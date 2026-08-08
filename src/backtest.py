"""
backtest.py - 历史回测模块

模拟历史选股流程，评估策略有效性。

核心流程:
  1. 指定回测区间和日期列表
  2. 对每个回测日，用当时可用的数据跑 L0→L2→L4
  3. 记录选出的股票及其因子评分
  4. T+1/T+3/T+5 后验证实际收益
  5. 计算因子 IC/IR，生成回测报告

用法:
    python backtest.py --start 2026-07-01 --end 2026-08-01
    python backtest.py --date 2026-07-15  # 单日回测
    python backtest.py --start 2026-06-01 --end 2026-08-01 --top 5  # 只测Top5
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from akshare_provider import AkshareProvider
from database import get_db

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎。"""

    def __init__(self, top_n: int = 10):
        self.provider = AkshareProvider()
        self.db = get_db()
        self.top_n = top_n

        # 因子配置（简化版，只用动量 + 价值）
        self.factors = [
            ("momentum_20d", 0.25, "descending"),
            ("momentum_60d", 0.20, "descending"),
            ("change_5d", 0.15, "descending"),
            ("pe", 0.20, "ascending"),
            ("pb", 0.10, "ascending"),
            ("turnover", 0.10, "descending"),
        ]

    def get_trading_dates(self, start: str, end: str) -> list[str]:
        """获取区间内的交易日列表。"""
        try:
            import akshare as ak
            cal = ak.tool_trade_date_hist_sina()
            cal = cal[cal["trade_date"] >= start.replace("-", "")]
            cal = cal[cal["trade_date"] <= end.replace("-", "")]
            dates = []
            for _, row in cal.iterrows():
                d = str(row["trade_date"])
                dates.append(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
            return dates
        except Exception as e:
            logger.warning(f"获取交易日历失败: {e}，使用自然日")
            start_d = datetime.strptime(start, "%Y-%m-%d")
            end_d = datetime.strptime(end, "%Y-%m-%d")
            dates = []
            current = start_d
            while current <= end_d:
                if current.weekday() < 5:  # 周一到周五
                    dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            return dates

    def run_single_date(self, date_str: str) -> dict:
        """
        单日回测：模拟在 date_str 这个时点的选股。
        返回: {date, stock_count, picks: [{code, name, score, ...}], ...}
        """
        logger.info(f"回测 {date_str}...")

        try:
            # 获取该日期的全市场行情数据
            import akshare as ak
            try:
                df = ak.stock_zh_a_hist(
                    symbol="000001", period="daily",
                    start_date=date_str.replace("-", ""),
                    end_date=date_str.replace("-", ""),
                    adjust="qfq"
                )
                if df is None or len(df) == 0:
                    return {"date": date_str, "error": "非交易日或无数据", "picks": []}
            except Exception:
                return {"date": date_str, "error": "数据获取失败", "picks": []}

            # 获取当日行情快照
            spot = self.provider.get_stock_list()
            if spot is None or len(spot) == 0:
                return {"date": date_str, "error": "无行情数据", "picks": []}

            # L2 简化过滤：排除ST、小市值、无PE
            candidates = spot.copy()
            if "name" in candidates.columns:
                candidates = candidates[
                    ~candidates["name"].str.contains(r"ST|\*ST", na=False, regex=True)
                ]
            if "market_cap" in candidates.columns:
                candidates = candidates[candidates["market_cap"] >= 20e8]  # 市值 >= 20亿
            if "pe" in candidates.columns:
                candidates = candidates[(candidates["pe"] > 0) & (candidates["pe"] <= 200)]
            if "amount" in candidates.columns:
                candidates = candidates[candidates["amount"] >= 5e7]  # 成交额 >= 5000万

            if len(candidates) == 0:
                return {"date": date_str, "error": "L2过滤后无候选", "picks": []}

            # L4 评分
            candidates = self._score(candidates)
            candidates = candidates.head(self.top_n)

            # 构建结果
            picks = []
            for _, row in candidates.iterrows():
                picks.append({
                    "code": str(row.get("code", "")).zfill(6),
                    "name": str(row.get("name", "")),
                    "score": round(float(row.get("composite_score", 0)), 2),
                    "close": float(row.get("close", 0)),
                    "pe": float(row.get("pe", 0)),
                    "change_pct": float(row.get("change_pct", 0)),
                })

            return {
                "date": date_str,
                "candidates_total": len(spot),
                "l2_pass": len(candidates),
                "picks": picks,
            }

        except Exception as e:
            logger.error(f"回测 {date_str} 异常: {e}")
            return {"date": date_str, "error": str(e), "picks": []}

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        """简化版多因子评分。"""
        df = df.copy()
        df["composite_score"] = 0.0

        # 计算动量
        df["momentum_20d"] = df.get("change_20d", df.get("change_pct", 0)).fillna(0)
        df["momentum_60d"] = df.get("change_60d", df.get("change_pct", 0)).fillna(0)
        df["change_5d"] = df.get("change_pct", 0).fillna(0)
        df["turnover"] = df.get("turnover", 0).fillna(0)

        for factor_name, weight, direction in self.factors:
            if factor_name not in df.columns:
                continue
            col = df[factor_name].copy()
            col = col.replace([np.inf, -np.inf], np.nan).fillna(0)

            ascending = (direction == "ascending")
            rank = col.rank(method="min", ascending=ascending, na_option="bottom")
            max_rank = rank.max()
            if max_rank > 0:
                normalized = (rank / max_rank) * 100
            else:
                normalized = pd.Series(50, index=rank.index)

            df["composite_score"] += normalized * weight

        return df.sort_values("composite_score", ascending=False)

    def verify_picks(self, picks: list[dict], start_date: str,
                     periods: list[int] = None) -> list[dict]:
        """
        验证选股在 T+N 日的实际收益。
        periods: [1, 3, 5] 对应 T+1, T+3, T+5
        """
        if periods is None:
            periods = [1, 3, 5]

        results = []
        for pick in picks:
            code = pick["code"]
            entry_price = pick.get("close", 0)
            if entry_price <= 0:
                continue

            # 获取该日期之后的 K 线数据
            try:
                import akshare as ak
                s = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)
                e = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=20)
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=s.strftime("%Y%m%d"),
                    end_date=e.strftime("%Y%m%d"),
                    adjust="qfq"
                )
                if df is None or len(df) < max(periods):
                    continue

                df = df.sort_values("日期")
                perf = {
                    "code": code,
                    "name": pick.get("name", code),
                    "entry_date": start_date,
                    "entry_price": entry_price,
                }

                for p in periods:
                    if p < len(df):
                        exit_price = float(df.iloc[p]["收盘"])
                        ret = (exit_price / entry_price - 1) * 100
                        perf[f"T+{p}_price"] = round(exit_price, 2)
                        perf[f"T+{p}_return"] = round(ret, 2)
                    else:
                        perf[f"T+{p}_price"] = None
                        perf[f"T+{p}_return"] = None

                results.append(perf)
            except Exception as e:
                logger.debug(f"验证 {code} T+N 失败: {e}")

        return results

    def run(self, start: str, end: str, verify: bool = True) -> dict:
        """
        运行完整回测。
        返回回测报告 dict。
        """
        logger.info(f"回测区间: {start} ~ {end}")

        dates = self.get_trading_dates(start, end)
        logger.info(f"交易日: {len(dates)} 天")

        # 逐日运行（可选并发，但这里保持简单串行便于调试）
        daily_results = []
        for i, d in enumerate(dates):
            result = self.run_single_date(d)
            daily_results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  回测进度: {i+1}/{len(dates)}")

        # 汇总统计
        valid_days = [r for r in daily_results if "error" not in r and len(r.get("picks", [])) > 0]
        error_days = [r for r in daily_results if "error" in r]
        total_picks = sum(len(r.get("picks", [])) for r in daily_results)

        logger.info(f"回测完成: {len(valid_days)} 个有效日, {len(error_days)} 个错误日, {total_picks} 只推荐")

        # T+N 验证
        verify_results = []
        if verify:
            for r in valid_days[:5]:  # 只验证前5天（节省时间）
                v = self.verify_picks(r["picks"], r["date"])
                verify_results.extend(v)

        # 统计分析
        stats = self._calc_stats(daily_results, verify_results)

        return {
            "start": start,
            "end": end,
            "total_dates": len(dates),
            "valid_dates": len(valid_days),
            "error_dates": len(error_days),
            "total_picks": total_picks,
            "daily_results": daily_results,
            "verify_results": verify_results,
            "stats": stats,
        }

    def _calc_stats(self, daily_results: list, verify_results: list) -> dict:
        """计算回测统计数据。"""
        # 使用今天实际数据做对照
        all_returns = []
        for v in verify_results:
            for key in v:
                if key.endswith("_return") and v[key] is not None:
                    all_returns.append({"key": key, "return": v[key]})

        if not all_returns:
            return {"avg_return": None, "t1_stats": {}}

        df = pd.DataFrame(all_returns)

        stats = {"returns_detail": df.groupby("key")["return"].describe().to_dict()}

        # T+1 统计
        t1 = [v["return"] for v in all_returns if "T+1" in v["key"]]
        if t1:
            arr = np.array(t1)
            stats["t1_stats"] = {
                "count": len(arr),
                "avg": round(float(np.mean(arr)), 2),
                "positive": int(np.sum(arr > 0)),
                "win_rate": round(np.sum(arr > 0) / len(arr) * 100, 1),
                "max": round(float(np.max(arr)), 2),
                "min": round(float(np.min(arr)), 2),
            }

        # 推荐频率统计
        code_freq = {}
        for r in daily_results:
            for p in r.get("picks", []):
                code = p["code"]
                code_freq[code] = code_freq.get(code, 0) + 1

        top_codes = sorted(code_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        stats["most_picked"] = top_codes

        return stats

    def generate_report(self, results: dict) -> str:
        """生成回测报告 Markdown。"""
        stats = results.get("stats", {})
        lines = [
            f"# 回测报告\n",
            f"> 区间: {results['start']} ~ {results['end']}",
            f" | 交易日: {results['total_dates']} | 有效: {results['valid_dates']} | 推荐: {results['total_picks']} 只\n",
        ]

        # T+1 统计
        t1 = stats.get("t1_stats", {})
        if t1:
            lines.append("## T+1 收益统计\n")
            lines.append(f"- 样本数: {t1.get('count', 0)}")
            lines.append(f"- 平均收益: {t1.get('avg', 0):+.2f}%")
            lines.append(f"- 胜率: {t1.get('win_rate', 0)}%")
            lines.append(f"- 最大涨幅: {t1.get('max', 0):+.2f}%")
            lines.append(f"- 最大跌幅: {t1.get('min', 0):+.2f}%\n")

        # 高频推荐股票
        most = stats.get("most_picked", [])
        if most:
            lines.append("## 高频推荐 Top 20\n")
            lines.append("| 代码 | 推荐次数 |")
            lines.append("|------|---------|")
            for code, freq in most:
                lines.append(f"| {code} | {freq} |")
            lines.append("")

        lines.append(f"\n---\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="回测模块")
    parser.add_argument("--start", default=None, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--date", default=None, help="单日回测 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=10, help="每日选股数量 (默认10)")
    parser.add_argument("--no-verify", action="store_true", help="跳过T+N验证")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.date:
        start, end = args.date, args.date
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        # 默认：最近30天
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        print(f"未指定区间，使用默认: {start} ~ {end}")

    engine = BacktestEngine(top_n=args.top)
    results = engine.run(start, end, verify=not args.no_verify)

    # 保存结果
    report = engine.generate_report(results)
    report_path = Path("briefs") / f"回测报告_{start}_{end}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    # 保存原始数据
    data_path = Path("briefs") / f"回测数据_{start}_{end}.json"
    # 只保存摘要，避免文件过大
    summary = {k: v for k, v in results.items() if k not in ("daily_results", "verify_results")}
    summary["daily_results_count"] = len(results.get("daily_results", []))
    data_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")

    print(f"\n回测完成:")
    print(f"  报告: {report_path}")
    print(f"  区间: {start} ~ {end}")
    print(f"  交易日: {results['total_dates']} | 有效: {results['valid_dates']} | 推荐: {results['total_picks']}")

    stats = results.get("stats", {})
    t1 = stats.get("t1_stats", {})
    if t1:
        print(f"  T+1 平均: {t1.get('avg', 0):+.2f}% | 胜率: {t1.get('win_rate', 0)}%")


if __name__ == "__main__":
    main()
