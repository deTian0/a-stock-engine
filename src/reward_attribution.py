"""
reward_attribution.py - 收益归因分析

分析持仓收益来源：
- 个股选择贡献（选股 alpha）
- 板块配置贡献（板块 beta）
- 市场择时贡献（仓位调整）

输出归因报告，帮助优化选股策略。
"""

import sys
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from westock_cli import get_cli, sector_of
from local_price_loader import LocalPriceLoader
from database import get_db

logger = logging.getLogger(__name__)


class RewardAttribution:
    """收益归因分析器。"""

    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.price_loader = LocalPriceLoader()
        self.cli = get_cli()

    def calc_attribution(self, holdings: dict, start_date: str,
                         end_date: str = None) -> dict:
        """
        计算持仓收益归因。

        Args:
            holdings: {code: {name, shares, cost_price}}
            start_date: 分析起始日期
            end_date: 分析结束日期（默认今天）

        Returns:
            {
                "total_return": float,          # 总收益率
                "stock_picking": float,         # 选股贡献
                "sector_allocation": float,     # 板块配置贡献
                "market_timing": float,         # 择时贡献
                "details": list[dict],          # 逐只股票明细
            }
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if not holdings:
            return {"total_return": 0, "stock_picking": 0,
                    "sector_allocation": 0, "market_timing": 0, "details": []}

        total_cost = 0
        total_value = 0
        details = []
        sector_returns = {}

        for code, info in holdings.items():
            name = info.get("name", code)
            shares = info.get("shares", 0)
            cost_price = info.get("cost_price", 0)

            try:
                df = self.price_loader.get_price(code, days=120)
                if len(df) < 5:
                    continue

                df["date"] = pd.to_datetime(df["date"])
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)

                start_mask = df["date"] >= start_ts
                end_mask = df["date"] <= end_ts

                if start_mask.sum() == 0 or end_mask.sum() == 0:
                    continue

                start_close = float(df[start_mask].iloc[0]["close"])
                end_close = float(df[end_mask].iloc[-1]["close"])

                cost_value = shares * cost_price
                current_value = shares * end_close
                hold_return = (end_close / cost_price - 1) * 100
                period_return = (end_close / start_close - 1) * 100

                total_cost += cost_value
                total_value += current_value

                stock_sector = sector_of(code)
                if stock_sector not in sector_returns:
                    sector_returns[stock_sector] = []
                sector_returns[stock_sector].append(period_return)

                details.append({
                    "code": code,
                    "name": name,
                    "sector": stock_sector,
                    "shares": shares,
                    "cost_price": cost_price,
                    "current_price": end_close,
                    "hold_return": round(hold_return, 2),
                    "period_return": round(period_return, 2),
                    "contribution": round((current_value - cost_value) / max(total_cost, 1) * 100, 2),
                })

            except Exception as e:
                logger.error(f"归因分析失败 {code}: {e}")

        total_return = round((total_value / max(total_cost, 1) - 1) * 100, 2) if total_cost > 0 else 0

        # 板块配置贡献（各板块平均收益的加权）
        sector_avg = {}
        for sector, returns in sector_returns.items():
            sector_avg[sector] = round(np.mean(returns), 2)
        sector_contribution = round(np.mean(list(sector_avg.values())), 2) if sector_avg else 0

        # 选股贡献（个股收益 - 板块平均收益）
        stock_picking = 0
        for d in details:
            s_avg = sector_avg.get(d["sector"], 0)
            stock_picking += d["period_return"] - s_avg
        stock_picking = round(stock_picking / max(len(details), 1), 2)

        return {
            "total_return": total_return,
            "stock_picking": stock_picking,
            "sector_allocation": sector_contribution,
            "market_timing": round(total_return - stock_picking - sector_contribution, 2),
            "details": details,
            "sector_breakdown": sector_avg,
        }

    def generate_report(self, attribution: dict, start_date: str,
                        end_date: str = None) -> str:
        """生成归因分析报告。"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        lines = [f"# 收益归因分析\n"]
        lines.append(f"> 分析区间: {start_date} ~ {end_date}\n")

        lines.append("## 归因汇总\n")
        lines.append(f"- **总收益率**: {attribution['total_return']:+.2f}%")
        lines.append(f"- **选股贡献** (alpha): {attribution['stock_picking']:+.2f}%")
        lines.append(f"- **板块配置贡献**: {attribution['sector_allocation']:+.2f}%")
        lines.append(f"- **择时贡献**: {attribution['market_timing']:+.2f}%\n")

        if attribution.get("sector_breakdown"):
            lines.append("## 板块收益分解\n")
            lines.append("| 板块 | 平均收益 |")
            lines.append("|------|---------|")
            for sector, ret in sorted(attribution["sector_breakdown"].items(),
                                       key=lambda x: x[1], reverse=True):
                emoji = "🔴" if ret > 0 else "🟢"
                lines.append(f"| {sector} | {emoji} {ret:+.2f}% |")
            lines.append("")

        if attribution.get("details"):
            lines.append("## 持仓明细\n")
            lines.append("| 代码 | 名称 | 板块 | 持仓收益 | 区间收益 | 贡献度 |")
            lines.append("|------|------|------|---------|---------|--------|")
            for d in sorted(attribution["details"], key=lambda x: x["period_return"], reverse=True):
                emoji = "🔴" if d["period_return"] > 0 else "🟢"
                lines.append(
                    f"| {d['code']} | {d['name']} | {d['sector']} | "
                    f"{d['hold_return']:+.2f}% | {emoji} {d['period_return']:+.2f}% | "
                    f"{d['contribution']:+.2f}% |"
                )

        lines.append(f"\n---\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    attributor = RewardAttribution()
    holdings = attributor.config.get("account", {}).get("holdings", {})

    if not holdings:
        print("请在 config.yaml 中配置 account.holdings 后运行")
        sys.exit(1)

    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    result = attributor.calc_attribution(holdings, start_date)
    report = attributor.generate_report(result, start_date)

    # 保存持仓快照到 SQLite
    try:
        db = get_db()
        # 获取当前价格信息构建快照
        snapshot_holdings = {}
        for d in result.get("details", []):
            code = d["code"]
            snapshot_holdings[code] = {
                "name": d["name"],
                "shares": d["shares"],
                "cost_price": d["cost_price"],
            }
        if snapshot_holdings:
            db.save_holdings_snapshot(snapshot_holdings)
    except Exception as e:
        logger.warning(f"持仓快照入库失败: {e}")

    report_path = Path("briefs") / datetime.now().strftime("%Y-%m-%d") / "收益归因分析.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"收益归因分析报告已生成: {report_path}")


if __name__ == "__main__":
    main()
