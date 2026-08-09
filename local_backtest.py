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
from database import get_db
from factor_engine import score_stocks, pick_top_by_sector, filter_candidates

logger = logging.getLogger(__name__)

# === CPU 安全参数 ===
DAY_SLEEP_SEC = 0.2       # 每日处理后的休眠
BATCH_SLEEP_SEC = 0.05    # 每 10 日批次后的休眠
MAX_PER_SECTOR = 5         # 每板块最大入选数


class LocalBacktest:
    """本地数据回测——SQLite Only。"""

    def __init__(self):
        self.db = get_db("data_cache/a-stock-engine.db")
        self.raw_conn = sqlite3.connect("data_cache/a-stock-engine.db")
        self._dates_cache: Optional[list[str]] = None
        self._day_data_cache: dict[str, pd.DataFrame] = {}  # 缓存最近读取的日期数据

    def get_available_dates(self) -> list[str]:
        """获取 DB 中所有可用日截面日期（带缓存）。"""
        if self._dates_cache is not None:
            return self._dates_cache
        c = self.raw_conn
        rows = c.execute(
            "SELECT DISTINCT cache_key FROM market_data_cache WHERE data_type='daily_snapshot' ORDER BY cache_key"
        ).fetchall()
        self._dates_cache = [r[0].replace("daily_snapshot_", "") for r in rows]
        logger.info(f"可用日期: {len(self._dates_cache)} 天 ({self._dates_cache[0]} ~ {self._dates_cache[-1]})")
        return self._dates_cache

    def load_day_data(self, date_str: str) -> pd.DataFrame:
        """从 SQLite 加载某日全量截面数据（带内存缓存）。"""
        if date_str in self._day_data_cache:
            return self._day_data_cache[date_str]
        cache_key = f"daily_snapshot_{date_str}"
        df = self.db.cache_get(cache_key)
        if df is not None and len(df) > 0:
            # 只缓存最近 40 天的数据（用于 T+N 验证）
            if len(self._day_data_cache) < 40:
                self._day_data_cache[date_str] = df
        return df if df is not None else pd.DataFrame()

    def filter_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """L2 过滤（委托 factor_engine）。"""
        return filter_candidates(df)

    def score_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """多因子评分（委托 factor_engine）。"""
        return score_stocks(df)

    def pick_by_sector(self, df: pd.DataFrame, max_per: int = 5) -> list[dict]:
        """按板块分组选 Top N（委托 factor_engine）。"""
        return pick_top_by_sector(df, max_per)

    @staticmethod
    def _zscore_within_group(df, col, direction=1):
        return pd.Series(0, index=df.index)  # no longer used, kept for compat

    def pick_by_sector(self, df: pd.DataFrame, max_per: int = 5) -> list[dict]:
        """按板块分组选 Top N。"""
        if "sector" not in df.columns:
            return self._top_n_raw(df, max_per * 10)

        picks = []
        for sector, group in df.groupby("sector"):
            top = group.head(max_per)
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                picks.append({
                    "sector": str(sector),
                    "code": str(row.get("code", "")),
                    "name": str(row.get("name", "")),
                    "score": round(float(row.get("composite_score", 0)), 2),
                    "close": float(row.get("close", 0)),
                    "pe": float(row.get("pe", 0)) if pd.notna(row.get("pe")) else 0,
                    "market_cap": float(row.get("market_cap", 0)) if pd.notna(row.get("market_cap")) else 0,
                    "rank": rank,
                })

        logger.debug(f"  {len(df['sector'].unique())} 板块, {len(picks)} 只选中")
        return picks

    def _top_n_raw(self, df: pd.DataFrame, n: int) -> list[dict]:
        """无板块信息时的兜底选股。"""
        picks = []
        for rank, (_, row) in enumerate(df.head(n).iterrows(), 1):
            picks.append({
                "sector": "未知",
                "code": str(row.get("code", "")),
                "name": str(row.get("name", "")),
                "score": round(float(row.get("composite_score", 0)), 2),
                "close": float(row.get("close", 0)),
                "pe": float(row.get("pe", 0)) if pd.notna(row.get("pe")) else 0,
                "market_cap": float(row.get("market_cap", 0)) if pd.notna(row.get("market_cap")) else 0,
                "rank": rank,
            })
        return picks

    def get_price_on_date(self, code: str, date_str: str) -> Optional[float]:
        """快速查询单只股票在某日的收盘价（直查 SQLite JSON）。"""
        c = self.raw_conn
        row = c.execute(
            "SELECT data_json FROM market_data_cache WHERE cache_key=?",
            (f"daily_snapshot_{date_str}",)
        ).fetchone()
        if row is None:
            return None
        # JSON 中搜索该 code 的 close 价（避免加载全量 DataFrame）
        import re
        pattern = f'"code":"{code}"'
        data = row[0]
        idx = data.find(pattern)
        if idx < 0:
            return None
        # 从匹配位置往后找 "close": <number>
        close_idx = data.find('"close":', idx)
        if close_idx < 0:
            return None
        end = data.find(',', close_idx)
        if end < 0:
            end = data.find('}', close_idx)
        if end < 0:
            return None
        val_str = data[close_idx+8:end]
        try:
            return float(val_str)
        except ValueError:
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

        for di, date_str in enumerate(all_dates):
            # CPU 安全
            if (di + 1) % 10 == 0:
                logger.info(f"  进度: {di+1}/{len(all_dates)}")
                time.sleep(BATCH_SLEEP_SEC)

            df = self.load_day_data(date_str)
            if len(df) == 0:
                continue

            df = self.filter_stocks(df)
            df = self.score_stocks(df)
            picks = self.pick_by_sector(df, MAX_PER_SECTOR)

            # 入库
            if picks:
                self.db.save_rotation_picks(
                    picks, date_str, date_str, "backtest"
                )
                for p in picks:
                    try:
                        self.db.upsert_pick_frequency(p["code"], "backtest")
                    except Exception:
                        pass

                daily_records.append({"date": date_str, "picks": picks, "filtered": len(df)})
                total_picks += len(picks)

            time.sleep(DAY_SLEEP_SEC)

        logger.info(f"回测完成: {len(daily_records)} 个有效日, {total_picks} 次推荐")

        # T+N 验证
        verify_results = []
        if verify and daily_records:
            verify_days = daily_records[-30:]  # 最后 30 天有足够 T+N 数据
            logger.info(f"验证 T+N 收益 ({len(verify_days)} 天)...")
            for rec in verify_days:
                v = self.verify_performance(rec["picks"], rec["date"])
                verify_results.extend(v)

        stats = self._compute_stats(verify_results)
        return {
            "total_dates": len(all_dates),
            "valid_dates": len(daily_records),
            "total_picks": total_picks,
            "stats": stats,
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

        # 输出报告
        print(f"\n{'='*60}")
        print(f"回测结果")
        print(f"{'='*60}")
        print(f"交易日: {results['total_dates']} | 有效日: {results['valid_dates']} | 推荐: {results['total_picks']} 次")

        for period in ["T+1_ret", "T+3_ret", "T+5_ret"]:
            if period in stats:
                s = stats[period]
                print(f"\n{period}:")
                print(f"  样本: {s['count']} | 平均: {s['avg']:+.2f}% | 胜率: {s['win_rate']}%")
                print(f"  最大: {s['max']:+.2f}% | 最小: {s['min']:+.2f}%")

        # 高频推荐
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
        bt.db.close()
        logger.info("DB 已关闭")


if __name__ == "__main__":
    main()
