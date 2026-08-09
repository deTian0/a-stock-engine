"""
sector_rotation_watchlist.py - 板块轮动监控

每天 16:00 收盘后运行，监控板块强弱轮动。
识别资金正在流入/流出的板块，为次日选股提供参考。

输出: briefs/YYYY-MM-DD/板块轮动监控.md
"""

import sys
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.westock import get_cli
from utils.guard import setup_protection, teardown_protection, setup_logging

logger = logging.getLogger(__name__)


class SectorRotationWatcher:
    """板块轮动监控器。"""

    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.cli = get_cli()

    def get_sector_data(self) -> pd.DataFrame:
        """获取各板块的行情数据。"""
        try:
            return self.cli.get_sector_list()
        except Exception as e:
            logger.error(f"获取板块数据失败: {e}")
            return pd.DataFrame()

    def calc_sector_strength(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算板块强度指标。
        - 近5日涨幅
        - 近20日涨幅
        - 成交额变化
        - 强度评分 = 短期涨幅 * 0.4 + 中期涨幅 * 0.4 + 成交额变化 * 0.2
        """
        if len(sector_df) == 0:
            return sector_df

        df = sector_df.copy()

        # 确保有必要的列
        required = ["name"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"板块数据缺少列: {col}")
                return df

        # 计算强度评分
        if "change_5d" in df.columns and "change_20d" in df.columns:
            df["strength_score"] = (
                df["change_5d"].fillna(0) * 0.4
                + df["change_20d"].fillna(0) * 0.4
                + (df.get("amount_change", pd.Series(0, index=df.index)).fillna(0)) * 0.2
            ).round(2)
            df = df.sort_values("strength_score", ascending=False)

        return df

    def identify_rotation(self, sector_df: pd.DataFrame) -> dict:
        """
        识别板块轮动信号。
        - 流入板块: 短期涨幅 > 0 且成交额放大
        - 流出板块: 短期涨幅 < 0 且成交额萎缩
        - 轮动信号: 从流出转向流入的板块
        """
        if len(sector_df) == 0:
            return {"inflow": [], "outflow": [], "rotation": []}

        inflow = []
        outflow = []

        for _, row in sector_df.iterrows():
            change_5d = row.get("change_5d", 0)
            amount_change = row.get("amount_change", 0)

            if pd.notna(change_5d) and pd.notna(amount_change):
                if change_5d > 2 and amount_change > 0.2:
                    inflow.append(row.get("name", ""))
                elif change_5d < -2 and amount_change < -0.2:
                    outflow.append(row.get("name", ""))

        return {
            "inflow": inflow[:10],   # 资金流入板块 Top 10
            "outflow": outflow[:10],  # 资金流出板块 Top 10
            "rotation": [],           # 轮动信号（需要历史数据对比）
        }

    def generate_report(self, sector_df: pd.DataFrame, rotation: dict) -> str:
        """生成板块轮动监控报告。"""
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [f"# 板块轮动监控 — {today}\n"]

        # 资金流入
        lines.append("## 资金流入板块\n")
        if rotation["inflow"]:
            for i, name in enumerate(rotation["inflow"], 1):
                lines.append(f"{i}. **{name}**")
        else:
            lines.append("_无明显流入_\n")

        # 资金流出
        lines.append("\n## 资金流出板块\n")
        if rotation["outflow"]:
            for i, name in enumerate(rotation["outflow"], 1):
                lines.append(f"{i}. **{name}**")
        else:
            lines.append("_无明显流出_\n")

        # 板块强度排行
        if "strength_score" in sector_df.columns:
            lines.append("\n## 板块强度排行（Top 20）\n")
            lines.append("| 排名 | 板块 | 近5日涨幅 | 近20日涨幅 | 强度评分 |")
            lines.append("|------|------|----------|-----------|----------|")
            for i, (_, row) in enumerate(sector_df.head(20).iterrows(), 1):
                c5 = f"{row.get('change_5d', 0):+.2f}%" if pd.notna(row.get("change_5d")) else "-"
                c20 = f"{row.get('change_20d', 0):+.2f}%" if pd.notna(row.get("change_20d")) else "-"
                score = row.get("strength_score", 0)
                lines.append(f"| {i} | {row.get('name', '-')} | {c5} | {c20} | {score} |")

        lines.append(f"\n---\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)


def main():
    setup_protection()
    setup_logging()

    try:
        watcher = SectorRotationWatcher()
        sector_df = watcher.get_sector_data()
        sector_df = watcher.calc_sector_strength(sector_df)
        rotation = watcher.identify_rotation(sector_df)
        report = watcher.generate_report(sector_df, rotation)

        today = datetime.now().strftime("%Y-%m-%d")
        report_path = Path("briefs") / today / "板块轮动监控.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        print(f"板块轮动监控报告已生成: {report_path}")

    except KeyboardInterrupt:
        logger.warning("用户中断，正在清理...")
    finally:
        teardown_protection()


if __name__ == "__main__":
    main()
