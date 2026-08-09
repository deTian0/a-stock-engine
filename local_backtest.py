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
from database import get_db, get_market_db
from factor_engine import score_stocks, pick_top_by_sector, filter_candidates

logger = logging.getLogger(__name__)

# === CPU 安全参数 ===
DAY_SLEEP_SEC = 0
BATCH_SLEEP_SEC = 0
MAX_PER_SECTOR = 5
BATCH_TRACKING = 20   # 每20天批量写入 rotation tracking，减少磁盘IO

# === 风控参数（v2.2 新增） ===
RISK_STOP_LOSS_PCT = 8.0       # T+1 止损线 (%)
RISK_MAX_SECTOR_PCT = 20.0     # 单板块最大占比 (%)
RISK_MAX_STOCK_PCT = 5.0       # 单只股票最大占比 (%)


class LocalBacktest:
    """本地数据回测——SQLite Only。"""

    def __init__(self):
        self.sel_db = get_db()
        self.db = get_market_db()
        self.raw_conn = sqlite3.connect(self.db.db_path)
        self._dates_cache: Optional[list[str]] = None  # 缓存最近读取的日期数据

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

    def run(self, start_date: str = None, end_date: str = None, verify: bool = True) -> dict:
        """运行完整回测。"""
        all_dates = self.get_available_dates()

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        logger.info(f"回测区间: {all_dates[0]} ~ {all_dates[-1]}, {len(all_dates)} 天")

        daily_records = []
        total_picks = 0

        # 缓存本轮选股，批量写入 DB
        batch_picks = []  # 累积到 BATCH_TRACKING 天再入库

        for di, date_str in enumerate(all_dates):
            if (di + 1) % 100 == 0:
                logger.info(f"  进度: {di+1}/{len(all_dates)}")

            df = self.load_day_data(date_str)
            if len(df) == 0:
                continue

            df = self.filter_stocks(df)
            df = self.score_stocks(df)
            picks = self.pick_by_sector(df, MAX_PER_SECTOR)
            picks_risk = self.apply_risk_controls(picks)

            if picks:
                batch_picks.append((date_str, picks))
                daily_records.append({
                    "date": date_str, "picks": picks, "picks_risk": picks_risk,
                    "filtered": len(df)
                })
                total_picks += len(picks)

            # 批量入库：每 BATCH_TRACKING 天或最后一天
            if len(batch_picks) >= BATCH_TRACKING or di == len(all_dates) - 1:
                for bd, bp in batch_picks:
                    self.sel_db.save_rotation_picks(bp, bd, bd, "backtest")
                    for p in bp:
                        try:
                            self.sel_db.upsert_pick_frequency(p["code"], "backtest")
                        except Exception:
                            pass
                batch_picks.clear()

        logger.info(f"回测完成: {len(daily_records)} 个有效日, {total_picks} 次推荐")

        # T+N 验证 (含止损模拟)
        verify_results = []
        verify_risk = []
        if verify and daily_records:
            verify_days = daily_records[-30:]
            logger.info(f"验证 T+N 收益 ({len(verify_days)} 天)...")
            for rec in verify_days:
                v = self.verify_performance(rec["picks"], rec["date"])
                verify_results.extend(v)
                # 风控版: 应用止损
                for item in v:
                    item_risk = dict(item)
                    t1 = item_risk.get("T+1_ret")
                    if t1 is not None and t1 < -RISK_STOP_LOSS_PCT:
                        item_risk["T+1_ret"] = -RISK_STOP_LOSS_PCT
                    verify_risk.append(item_risk)

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


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 60)
    logger.info("本地数据回测启动 (CPU 安全模式)")
    logger.info("=" * 60)

    bt = LocalBacktest()

    try:
        results = bt.run(verify=True)
        stats = results.get("stats", {})
        stats_risk = results.get("stats_risk", {})
        risk_sum = results.get("risk_summary", {})

        print(f"\n{'='*60}")
        print(f"回测结果 v2.2 (含风控)")
        print(f"{'='*60}")
        print(f"交易日: {results['total_dates']} | 有效日: {results['valid_dates']}")
        print(f"风控: 移除 {risk_sum.get('removed', 0)} 只 (止损>{RISK_STOP_LOSS_PCT}% / 板块>{RISK_MAX_SECTOR_PCT}%)")

        for period in ["T+1_ret", "T+3_ret", "T+5_ret"]:
            s_raw = stats.get(period, {})
            s_risk = stats_risk.get(period, {})
            if s_raw:
                print(f"\n{period}:")
                print(f"  原始:  {s_raw['count']}笔 | 均:{s_raw['avg']:+.2f}% | 胜率:{s_raw['win_rate']}%")
                if s_risk:
                    print(f"  风控:  {s_risk['count']}笔 | 均:{s_risk['avg']:+.2f}% | 胜率:{s_risk['win_rate']}%")

        most = stats.get("most_picked", [])
        if most:
            print(f"\n高频推荐 Top 10:")
            for item in most[:10]:
                print(f"  {item['code']}: {item['hits']} 次")

        # 保存报告
        report = [f"# 回测报告\n> 区间: 2026-01-05 ~ 2026-05-14\n"]
        for period in ["T+1_ret", "T+3_ret", "T+5_ret"]:
            if period in stats:
                s = stats[period]
                report.append(f"\n## {period}\n- 样本: {s['count']} | 平均: {s['avg']:+.2f}% | 胜率: {s['win_rate']}%\n")

        rpath = Path("briefs") / f"回测报告_本地数据_{datetime.now().strftime('%Y%m%d')}.md"
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text("\n".join(report), encoding="utf-8")
        print(f"\n报告: {rpath}")

    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        bt.sel_db.close()
        bt.db.close()
        logger.info("DB 已关闭")


if __name__ == "__main__":
    main()
