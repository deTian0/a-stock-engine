"""
local_backtest.py — 基于本地 SQLite 数据的回测引擎 (CPU 安全版)

特点:
  - 纯本地 SQLite 读取，零网络调用
  - 全串行处理，单文件单日逐一推进
  - 因子评分使用纯 pandas 向量化运算
  - 板块轮动: 每板块 Top5，两周滑动窗口
  - T+N 验证 (1/3/5 日)
  - CPU 保护: 每日处理间隔 sleep(0.2s)

用法:
    python local_backtest.py
"""

import sys, os, time, logging, sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from database import get_market_db
from factor_engine import score_stocks, pick_top_by_sector, filter_candidates

logger = logging.getLogger(__name__)

# === CPU 安全参数 ===
DAY_SLEEP_SEC = 0
BATCH_SLEEP_SEC = 0
MAX_PER_SECTOR = 5

# === 风控参数（v2.2 新增） ===
RISK_STOP_LOSS_PCT = 8.0       # T+1 止损线 (%)
RISK_MAX_SECTOR_PCT = 20.0     # 单板块最大占比 (%)
RISK_MAX_STOCK_PCT = 5.0
T_PERIODS = [1, 3, 5]
MARKET_MA = 60
INITIAL_CAPITAL = 50000
MAX_PICKS_PER_DAY = 20
TRADE_COST = 0.0013          # 0.13% 交易成本 (股票: 万0.854+印花税)
ETF_COST = 0.00017           # 0.017% ETF交易成本 (免印花税)
STOP_LOSS = 5.0              # 止损 (%)
TARGET_BASE = 3.0            # 基础目标收益 (%)       # 单只股票最大占比 (%)


class LocalBacktest:
    """本地数据回测——SQLite Only。"""

    def __init__(self):
        self.db = get_market_db()
        self.raw_conn = sqlite3.connect(self.db.db_path)
        self._dates_cache: Optional[list[str]] = None
        self._survivors: Optional[set] = None  # 存活股票集合

    def _compute_survivors(self, min_days: int = 252) -> set:
        """幸存者偏差校正：排除上市<1年的新股和即将退市的。"""
        if self._survivors is not None:
            return self._survivors
        c = self.raw_conn
        rows = c.execute(
            "SELECT code, COUNT(*) as cnt FROM daily_price GROUP BY code HAVING cnt >= ?",
            (min_days,)
        ).fetchall()
        self._survivors = {r[0] for r in rows}
        logger.info(f"幸存者过滤: {len(self._survivors)} 只 (>={min_days}天)")
        return self._survivors  # 缓存最近读取的日期数据

    def get_available_dates(self) -> list[str]:
        """获取所有可用交易日（后复权CSV + parquet 合并）。"""
        if self._dates_cache is not None:
            return self._dates_cache
        c = self.raw_conn
        rows = c.execute("SELECT DISTINCT date FROM daily_price ORDER BY date").fetchall()
        self._dates_cache = [r[0] for r in rows if r[0]]
        logger.info(f"可用日期: {len(self._dates_cache)} 天 ({self._dates_cache[0]} ~ {self._dates_cache[-1]})")
        return self._dates_cache

    def load_day_data(self, date_str: str) -> pd.DataFrame:
        """加载截面：parquet价格 + 后复权基本面合并（无缓存）。"""
        # 从 daily_price 读价格
        c = self.raw_conn
        rows = c.execute(
            "SELECT code, close, pct_chg, vol, amount FROM daily_price WHERE date=?",
            (date_str,)
        ).fetchall()
        if not rows:
            return pd.DataFrame()

        # 幸存者过滤
        survivors = self._compute_survivors()
        rows = [r for r in rows if r[0] in survivors]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["code_raw", "close", "pct_chg", "vol", "amount"])
        # 清理后缀用于匹配基本面
        df["code"] = df["code_raw"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
        df["sector"] = "其他"
        df["name"] = df["code"]

        # 合并基本面
        try:
            fund_df = self.db.cache_get(f"daily_snapshot_{date_str}")
            if fund_df is not None and len(fund_df) > 0:
                fund_df["code"] = fund_df["code"].astype(str).str.zfill(6)
                for col in ["name", "sector", "pe", "pb", "market_cap",
                            "turnover", "amplitude",
                            "chg_3d", "chg_6d", "chg_10d", "chg_25d", "change_pct"]:
                    if col in fund_df.columns:
                        fund_map = fund_df.set_index("code")[col].to_dict()
                        df[col] = df["code"].map(fund_map)
        except Exception as e:
            pass

        return df

    def filter_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """L2 过滤：有基本面 → factor_engine，纯价格 → 简化过滤。"""
        has_fundamentals = "pe" in df.columns and df["pe"].notna().sum() > 100
        if has_fundamentals:
            return filter_candidates(df)
        # 纯价格过滤：排除价格异常
        c = df.copy()
        if "close" in c.columns:
            c = c[c["close"] > 0]
        if "pct_chg" in c.columns:
            c = c[c["pct_chg"] < 20]  # 排除涨停
        return c

    def score_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """评分：有基本面→factor_engine，纯价格→动量因子。"""
        has_fundamentals = "market_cap" in df.columns and df["market_cap"].notna().sum() > 100
        if has_fundamentals:
            return score_stocks(df)
        # 纯动量评分
        s = df.copy()
        if "pct_chg" in s.columns:
            s["composite_score"] = s["pct_chg"].fillna(0)
        else:
            s["composite_score"] = 0
        return s.sort_values("composite_score", ascending=False)

    def pick_by_sector(self, df: pd.DataFrame, max_per: int = 5) -> list[dict]:
        """按板块分组选 Top N（委托 factor_engine）。"""
        return pick_top_by_sector(df, max_per)

    @staticmethod
    def _zscore_within_group(df, col, direction=1):
        return pd.Series(0, index=df.index)  # no longer used, kept for compat

    def pick_by_sector(self, df: pd.DataFrame, max_per: int = 5) -> list[dict]:
        """按板块分组选 Top N（委托 factor_engine）。"""
        return pick_top_by_sector(df, max_per)

    def apply_risk_controls(self, picks: list[dict]) -> list[dict]:
        """风控过滤（v2.2）：板块集中度 + 单股权重限制。"""
        if not picks:
            return picks
        total = len(picks)
        max_per_sector = int(total * RISK_MAX_SECTOR_PCT / 100)

        # 统计板块计数
        sector_counts = {}
        filtered = []
        for p in picks:
            s = p.get("sector", "未知")
            cnt = sector_counts.get(s, 0)
            if cnt >= max_per_sector:
                continue  # 板块超限，跳过
            if cnt >= total * RISK_MAX_STOCK_PCT / 100 * 10:
                continue  # 单股超限
            sector_counts[s] = cnt + 1
            filtered.append(p)

        return filtered

    def get_price_on_date(self, code: str, date_str: str) -> Optional[float]:
        """查询股票收盘价（自动补后缀）。"""
        c = self.raw_conn
        # 尝试无后缀、.SZ、.SH、.BJ
        for suffix in ["", ".SZ", ".SH", ".BJ"]:
            lookup = code + suffix
            row = c.execute(
                "SELECT close FROM daily_price WHERE code=? AND date=?",
                (lookup, date_str)
            ).fetchone()
            if row:
                return row[0]
        return None

    def verify_performance(self, picks: list[dict], entry_date: str, periods: list[int] = None) -> list[dict]:
        """T+N 收益验证（直查 SQL，不加载全天数据）。"""
        if periods is None:
            periods = [1, 3, 5]

        all_dates = self.get_available_dates()
        date_idx = None
        for i, d in enumerate(all_dates):
            if d >= entry_date:
                date_idx = i
                break
        if date_idx is None:
            return [dict(p, **{f"T+{p}_ret": None for p in periods}) for p in picks]

        results = []
        for pick in picks:
            code = pick["code"]
            entry_price = pick["close"]
            perf = dict(pick)

            for p in periods:
                target_idx = date_idx + p
                if target_idx >= len(all_dates):
                    perf[f"T+{p}_ret"] = None
                    continue

                exit_price = self.get_price_on_date(code, all_dates[target_idx])
                if exit_price and entry_price and entry_price > 0:
                    perf[f"T+{p}_ret"] = round((exit_price / entry_price - 1) * 100, 2)
                else:
                    perf[f"T+{p}_ret"] = None

            results.append(perf)

        return results

    def run(self, start_date: str = None, end_date: str = None) -> dict:
        """运行回测（纯计算，不写DB）。"""
        all_dates = self.get_available_dates()
        keep_last = T_PERIODS[-1] + 5  # 只保留 T+N 验证所需天数

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        logger.info(f"回测区间: {all_dates[0]} ~ {all_dates[-1]}, {len(all_dates)} 天")

        # === 预计算市场择时信号 (MA60) ===
        logger.info(f"计算择时信号 (MA{MARKET_MA})...")
        c = self.raw_conn
        market_avg = {}
        for i, d in enumerate(all_dates):
            r = c.execute("SELECT AVG(close) FROM daily_price WHERE date=?", (d,)).fetchone()
            market_avg[d] = r[0] if r[0] else 0
            if (i + 1) % 500 == 0:
                logger.info(f"  均价: {i+1}/{len(all_dates)}")

        avg_vals = [market_avg[d] for d in all_dates]
        ma60 = [sum(avg_vals[max(0, i - MARKET_MA + 1):i + 1]) / min(i + 1, MARKET_MA)
                for i in range(len(all_dates))]
        market_regime = {all_dates[i]: avg_vals[i] > ma60[i] for i in range(len(all_dates))}
        trade_days = sum(market_regime.values())
        logger.info(f"择时: {trade_days}/{len(all_dates)} 天可交易 ({trade_days/len(all_dates)*100:.0f}%)")

        daily_records = []
        total_picks = 0

        # 缓存本轮选股，批量写入 DB

        # 边跑边验：当天选股如果 T+N 数据已就绪，即时验证
        verify_results = []
        verify_risk = []
        max_period = max(T_PERIODS)
        date_index = {d: i for i, d in enumerate(all_dates)}  # O(1) lookup

        for di, date_str in enumerate(all_dates):
            if (di + 1) % 100 == 0:
                logger.info(f"  进度: {di+1}/{len(all_dates)}")

            # 择时：市场在 MA60 以下 → 空仓
            if not market_regime.get(date_str, True):
                continue

            df = self.load_day_data(date_str)
            if len(df) == 0:
                continue

            df = self.filter_stocks(df)
            df = self.score_stocks(df)
            picks = self.pick_by_sector(df, MAX_PER_SECTOR)
            picks_risk = self.apply_risk_controls(picks)

            if picks:
                daily_records.append({
                    "date": date_str, "picks": picks, "picks_risk": picks_risk,
                    "filtered": len(df)
                })
                total_picks += len(picks)

            # 即时验证：往前找最早可验证的日期（T+N数据已就绪）
            for rec in list(daily_records):  # 遍历副本
                if date_str >= rec["date"]:  # 还没到未来，跳过
                    idx_then = date_index[rec["date"]]
                    if idx_now - idx_then >= max_period:
                        v = self.verify_performance(rec["picks"], rec["date"])
                        verify_results.extend(v)
                        for item in v:
                            item_risk = dict(item)
                            t1 = item_risk.get("T+1_ret")
                            if t1 is not None and t1 < -RISK_STOP_LOSS_PCT:
                                item_risk["T+1_ret"] = -RISK_STOP_LOSS_PCT
                            verify_risk.append(item_risk)
                        daily_records.remove(rec)  # 已验，移除

        logger.info(f"回测完成: {len(daily_records)} 个未验日, {total_picks} 次推荐, {len(verify_results)} 次验证")

        stats = self._compute_stats(verify_results)
        stats_risk = self._compute_stats(verify_risk)

        # 风控统计摘要
        raw_picks = sum(len(r["picks"]) for r in daily_records)
        risk_picks = sum(len(r.get("picks_risk", r["picks"])) for r in daily_records)

        return {
            "total_dates": len(all_dates),
            "valid_dates": len(daily_records),
            "total_picks": total_picks,
            "stats": stats,
            "stats_risk": stats_risk,
            "risk_summary": {
                "before_risk": raw_picks,
                "after_risk": risk_picks,
                "removed": raw_picks - risk_picks,
                "stop_loss": RISK_STOP_LOSS_PCT,
                "max_sector_pct": RISK_MAX_SECTOR_PCT,
                "max_stock_pct": RISK_MAX_STOCK_PCT,
            },
            "verify_results": verify_results,
        }

    def _compute_stats(self, verify_results: list) -> dict:
        """计算统计指标。"""
        all_ret = []
        for v in verify_results:
            for key in ["T+1_ret", "T+3_ret", "T+5_ret"]:
                if key in v and v[key] is not None:
                    all_ret.append({"key": key, "return": v[key]})

        if not all_ret:
            return {}

        df_ret = pd.DataFrame(all_ret)
        stats = {}
        for period in ["T+1_ret", "T+3_ret", "T+5_ret"]:
            vals = [r["return"] for r in all_ret if r["key"] == period]
            if vals:
                arr = np.array(vals)
                stats[period] = {
                    "count": len(arr),
                    "avg": round(float(np.mean(arr)), 2),
                    "win_rate": round(float(np.sum(arr > 0)) / len(arr) * 100, 1),
                    "max": round(float(np.max(arr)), 2),
                    "min": round(float(np.min(arr)), 2),
                }

        # 高频推荐
        code_freq = {}
        for v in verify_results:
            code = v.get("code", "")
            code_freq[code] = code_freq.get(code, 0) + 1
        top_codes = sorted(code_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        stats["most_picked"] = [
            {"code": c, "name": "", "hits": f} for c, f in top_codes
        ]

        return stats


    @staticmethod
    def _trade_cost(code: str, is_buy: bool = True) -> float:
        """计算交易成本率。ETF 免印花税。"""
        if code.startswith(("15", "51", "56", "58", "510", "512", "513", "515", "516",
                            "517", "518", "560", "561", "562", "563", "588")):
            return ETF_COST  # ETF 只有佣金，无印花税
        return TRADE_COST

    def run_portfolio(self) -> dict:
        """资金模拟：5万起，每只有目标/止损，动态持仓周期。"""
        all_dates = self.get_available_dates()
        c = self.raw_conn

        # 市场择时
        market_avg = {}
        for d in all_dates:
            r = c.execute("SELECT AVG(close) FROM daily_price WHERE date=?", (d,)).fetchone()
            market_avg[d] = r[0] or 0
        avg_vals = [market_avg[d] for d in all_dates]
        ma60 = [sum(avg_vals[max(0,i-MARKET_MA+1):i+1])/min(i+1,MARKET_MA) for i in range(len(all_dates))]
        regime = {all_dates[i]: avg_vals[i] > ma60[i] for i in range(len(all_dates))}

        cash = float(INITIAL_CAPITAL)
        positions = {}  # {code: {buy_price, shares, target, stop, hold_days, entry_idx, score}}
        portfolio = []
        sell_log = []   # 记录每笔卖出: (code, buy_p, sell_p, held, ret%, reason)

        logger.info(f"资金模拟: 初始{cash:,.0f}元, 日选{MAX_PICKS_PER_DAY}只")
        logger.info(f"规则: 目标+{TARGET_BASE}~8%, 止损-{STOP_LOSS}%, 成本{TRADE_COST*100:.1f}%")

        for di, date_str in enumerate(all_dates):
            if (di + 1) % 200 == 0:
                logger.info(f"  进度: {di+1}/{len(all_dates)} | 持仓{len(positions)} | 现金{cash:,.0f}")

            # === 每日持仓评估 ===
            to_sell = []
            for code, pos in list(positions.items()):
                cur_price = self.get_price_on_date(code, date_str)
                if cur_price is None:
                    cur_price = pos["buy_price"]
                held_days = di - pos["entry_idx"]
                ret_pct = (cur_price / pos["buy_price"] - 1) * 100
                reason = None

                # 1. 触碰止损
                if ret_pct <= -STOP_LOSS:
                    reason = f"止损({ret_pct:+.1f}%)"
                # 2. 达到目标
                elif cur_price >= pos["target"]:
                    reason = f"达到目标({ret_pct:+.1f}%)"
                # 3. 持仓到期
                elif held_days >= pos["hold_days"]:
                    reason = f"持仓到期({held_days}天)"
                # 4. 动量衰减（跌破买入价且持有>3天）
                elif held_days > 3 and cur_price < pos["buy_price"]:
                    reason = f"动量衰减({ret_pct:+.1f}%)"

                if reason:
                    cash += pos["shares"] * cur_price * (1 - self._trade_cost(code))  # 卖出
                    sell_log.append({
                        "code": code, "buy_p": pos["buy_price"], "sell_p": cur_price,
                        "held": held_days, "ret": round(ret_pct, 2), "reason": reason
                    })
                    to_sell.append(code)

            for code in to_sell:
                del positions[code]

            # === 计算总资产 ===
            total = cash
            for pos in positions.values():
                cp = self.get_price_on_date(pos["code"], date_str) or pos["buy_price"]
                total += pos["shares"] * cp
            portfolio.append((date_str, round(total, 2)))

            # === 入场判断 ===
            if not regime.get(date_str, True):
                continue
            if cash < 5000:  # 资金太少不买
                continue

            df = self.load_day_data(date_str)
            if len(df) == 0:
                continue
            df = self.filter_stocks(df)
            df = self.score_stocks(df)

            # Top N, 排除已有持仓
            top = df[~df["code"].isin(set(positions.keys()))].head(MAX_PICKS_PER_DAY * 3)
            if len(top) == 0:
                continue

            buy_count = 0
            scores = top["composite_score"].values
            s_median = float(np.median(scores[scores == scores])) if len(scores) > 0 else 0
            s_max = float(np.max(scores)) if len(scores) > 0 else 1
            s_range = max(s_max - s_median, 0.01)

            for _, row in top.iterrows():
                code = row["code"]
                price = float(row["close"]) if pd.notna(row.get("close")) else 0
                if price <= 0:
                    continue

                alloc = cash / max(MAX_PICKS_PER_DAY - buy_count, 1)
                shares = int(alloc / price / 100) * 100
                if shares < 100:
                    continue

                cost = shares * price * (1 + self._trade_cost(code, is_buy=True))  # 买入
                if cost > cash * 0.5:
                    continue

                # 动态持仓天数：分高→多拿，分低→早走
                s = float(row["composite_score"])
                percentile = min(1.0, max(0.1, (s - s_median) / s_range + 0.5))
                hold_days = max(3, min(10, int(5 + percentile * 5)))  # 3-10天

                # 目标价：分越高目标越高
                target_pct = TARGET_BASE + percentile * 5  # 3%-8%
                target_price = price * (1 + target_pct / 100)
                stop_price = price * (1 - STOP_LOSS / 100)

                cash -= cost
                positions[code] = {
                    "code": code, "buy_price": price, "shares": shares,
                    "target": round(target_price, 2), "stop": round(stop_price, 2),
                    "hold_days": hold_days, "entry_idx": di, "score": round(s, 2)
                }
                buy_count += 1
                if buy_count >= MAX_PICKS_PER_DAY:
                    break

        # 清仓
        last_date = all_dates[-1]
        for code, pos in list(positions.items()):
            sp = self.get_price_on_date(code, last_date) or pos["buy_price"]
            cash += pos["shares"] * sp * (1 - self._trade_cost(code))  # 清仓
        total = cash
        portfolio.append((last_date, round(total, 2)))

        # === 统计 ===
        values = [v for _, v in portfolio]
        returns_daily = [(values[i]/values[i-1]-1) for i in range(1,len(values)) if values[i-1]>0]
        years = len(values) / 252
        cagr = (values[-1]/INITIAL_CAPITAL)**(1/years)-1 if years>0 else 0

        peak = values[0]; max_dd = 0.0
        for v in values:
            if v > peak: peak = v
            dd = (peak-v)/peak if peak>0 else 0
            if dd > max_dd: max_dd = dd

        ret_arr = np.array(returns_daily) if returns_daily else np.array([0])
        rf_daily = 0.02/252
        excess = ret_arr - rf_daily
        sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess)>0 else 0

        # 卖出原因统计
        reasons = {}
        for sl in sell_log:
            r = sl["reason"].split("(")[0]
            reasons[r] = reasons.get(r, 0) + 1
        avg_held = np.mean([sl["held"] for sl in sell_log]) if sell_log else 0
        avg_ret = np.mean([sl["ret"] for sl in sell_log]) if sell_log else 0

        # 逐年
        yearly = {}
        for d, v in portfolio:
            y = d[:4]; yearly.setdefault(y, []).append(v)
        year_returns = {y: round((vals[-1]/vals[0]-1)*100,1) for y,vals in yearly.items() if vals[0]>0}

        return {
            "initial": INITIAL_CAPITAL, "final": round(values[-1],2),
            "return_pct": round((values[-1]/INITIAL_CAPITAL-1)*100,2),
            "cagr_pct": round(cagr*100,2), "max_drawdown_pct": round(max_dd*100,2),
            "sharpe": round(sharpe,2), "years": round(years,1),
            "year_returns": year_returns, "portfolio": portfolio,
            "sell_stats": {"reasons": reasons, "avg_held_days": round(avg_held,1),
                           "avg_return": round(avg_ret,2), "total_trades": len(sell_log)},
        }


def main():
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 60)
    logger.info("本地数据回测启动")
    logger.info("=" * 60)

    bt = LocalBacktest()

    try:
        # === 资金模拟 ===
        logger.info("\n" + "=" * 60)
        logger.info(f"资金模拟: 初始 {INITIAL_CAPITAL:,} 元, 日选 {MAX_PICKS_PER_DAY} 只, 动态持仓 3-10天")
        logger.info("=" * 60)
        pf = bt.run_portfolio()

        print(f"\n{'='*60}")
        print(f"资金模拟结果 ({pf['years']}年)")
        print(f"{'='*60}")
        print(f"初始资金: ¥{pf['initial']:,.0f}")
        print(f"最终资金: ¥{pf['final']:,.0f}")
        print(f"总收益:   {pf['return_pct']:+.2f}%")
        print(f"年化收益: {pf['cagr_pct']:+.2f}%")
        print(f"最大回撤: {pf['max_drawdown_pct']:.1f}%")
        print(f"夏普比率: {pf['sharpe']}")

        if pf.get("year_returns"):
            print(f"\n逐年收益:")
            for y, r in sorted(pf["year_returns"].items()):
                print(f"  {y}: {r:+.1f}%")

        # 卖出统计
        sell_stats = pf.get("sell_stats", {})
        if sell_stats:
            print(f"\n卖出统计 ({sell_stats['total_trades']}笔):")
            print(f"  平均持有: {sell_stats['avg_held_days']}天 | 平均收益: {sell_stats['avg_return']:+.2f}%")
            for reason, cnt in sell_stats.get("reasons", {}).items():
                print(f"  {reason}: {cnt}笔")

        # 保存资产曲线
        portfolio = pf.get("portfolio", [])
        if portfolio:
            csv_path = Path("briefs") / f"portfolio_curve_{datetime.now().strftime('%Y%m%d')}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(portfolio, columns=["date", "value"]).to_csv(csv_path, index=False)
            print(f"\n资产曲线: {csv_path}")

    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        bt.db.close()
        logger.info("DB 已关闭")


if __name__ == "__main__":
    main()
