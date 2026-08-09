"""
daily_brief.py - 每日盘前选股简报生成器

这是自动化任务的入口点，每天 09:00 执行。
调用 multifactor.py 引擎，生成 Markdown 格式的盘前简报。

输出: briefs/YYYY-MM-DD/盘前选股简报.md
"""

import sys
import os
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# 确保能 import src 下的子包
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/

from engine.selection import MultiFactorEngine
from data.db import get_db
from utils.guard import setup_protection, teardown_protection, setup_logging

logger = logging.getLogger(__name__)


def generate_brief(results: dict, config: dict) -> str:
    """
    根据引擎结果生成 Markdown 格式的盘前简报。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    regime = results["regime"]
    categories = results["categories"]

    lines = []
    lines.append(f"# 盘前选股简报 — {today}\n")
    lines.append(f"> 生成时间: {results['timestamp']} | 耗时: {results['elapsed_seconds']}s\n")

    # ---- 市场环境 ----
    lines.append("## 一、市场环境判断\n")
    lines.append(f"**当前环境: {regime['regime']}** | 仓位上限: {regime['position_cap']:.0%}\n")
    lines.append(f"_{regime['judgment']}_\n")
    lines.append("\n| 指数 | 收盘 | MA20 | MA60 | 站上MA60 |")
    lines.append("|------|------|------|------|----------|")
    for code, info in regime["indices"].items():
        if "error" in info:
            lines.append(f"| {info['name']} | 数据获取失败 | - | - | - |")
        else:
            above = "✅" if info.get("above_ma") else "❌"
            lines.append(
                f"| {info['name']} | {info.get('close', '-')} | "
                f"{info.get('ma_short', '-')} | {info.get('ma_long', '-')} | {above} |"
            )
    lines.append("")

    # ---- ②A 质量榜 ----
    quality = categories.get("②A_质量榜")
    lines.append(f"\n## 二、②A 质量榜（Top {len(quality) if quality is not None else 0}）\n")
    if quality is not None and len(quality) > 0:
        lines.append("| 代码 | 名称 | 板块 | 综合评分 | ROE | 营收增速 | 动量20日 |")
        lines.append("|------|------|------|----------|-----|----------|----------|")
        for _, row in quality.iterrows():
            name = row.get("name", row["code"])
            sector = row.get("sector", "-")
            score = row.get("composite_score", 0)
            roe = f"{row.get('roe', 0):.1f}%" if pd.notna(row.get("roe")) else "-"
            rev_g = f"{row.get('revenue_growth', 0):.1f}%" if pd.notna(row.get("revenue_growth")) else "-"
            mom = f"{row.get('momentum_20d', 0):.1f}%" if pd.notna(row.get("momentum_20d")) else "-"
            lines.append(f"| {row['code']} | {name} | {sector} | {score} | {roe} | {rev_g} | {mom} |")
    else:
        lines.append("_今日无入选_\n")

    # ---- ②B 短线榜 ----
    short_list = categories.get("②B_短线榜")
    lines.append(f"\n## 三、②B 短线榜（{len(short_list) if short_list is not None else 0} 只）\n")
    if short_list is not None and len(short_list) > 0:
        if "decline_10d" in short_list.columns:
            # 反弹引擎结果
            lines.append("| 代码 | 名称 | 板块 | 近10日跌幅 | 量比 | 反弹涨幅 |")
            lines.append("|------|------|------|-----------|------|----------|")
            for _, row in short_list.iterrows():
                lines.append(
                    f"| {row['code']} | {row.get('name', row['code'])} | "
                    f"{row.get('sector', '-')} | {row.get('decline_10d', '-')}% | "
                    f"{row.get('volume_ratio', '-')} | {row.get('bounce_return', '-')}% |"
                )
        else:
            lines.append("| 代码 | 名称 | 板块 | 动量20日 | 动量60日 |")
            lines.append("|------|------|------|---------|---------|")
            for _, row in short_list.iterrows():
                lines.append(
                    f"| {row['code']} | {row.get('name', row['code'])} | "
                    f"{row.get('sector', '-')} | {row.get('momentum_20d', 0):.1f}% | "
                    f"{row.get('momentum_60d', 0):.1f}% |"
                )
    else:
        lines.append("_今日无短线候选_\n")

    # ---- ③A 持仓 ----
    holdings = categories.get("③A_持仓")
    lines.append(f"\n## 四、③A 当前持仓（{len(holdings) if holdings is not None else 0} 只）\n")
    if holdings is not None and len(holdings) > 0:
        lines.append("| 代码 | 名称 | 综合评分 | 板块 | 建议 |")
        lines.append("|------|------|----------|------|------|")
        for _, row in holdings.iterrows():
            score = row.get("composite_score", 0)
            l4 = results.get("l4_results", pd.DataFrame())
            if len(l4) > 0 and "composite_score" in l4.columns:
                median_val = l4["composite_score"].median()
                if pd.isna(median_val):
                    median_val = 50
            else:
                median_val = 50
            median = median_val
            advice = "⚠️ 关注" if score < median else "✅ 持有"
            lines.append(
                f"| {row['code']} | {row.get('name', row['code'])} | "
                f"{score} | {row.get('sector', '-')} | {advice} |"
            )
    else:
        lines.append("_当前无持仓（请在 config.yaml 中配置 holdings）_\n")

    # ---- ③B 操作建议 ----
    sells = categories.get("③B_操作建议")
    if sells is not None and len(sells) > 0:
        lines.append(f"\n## 五、③B 操作建议（{len(sells)} 只建议关注）\n")
        lines.append("| 代码 | 名称 | 综合评分 | 建议 |")
        lines.append("|------|------|----------|------|")
        for _, row in sells.iterrows():
            lines.append(
                f"| {row['code']} | {row.get('name', row['code'])} | "
                f"{row.get('composite_score', 0)} | 考虑减仓 |"
            )

    # ---- ③C 观察名单 ----
    watchlist = categories.get("③C_观察名单")
    lines.append(f"\n## 六、③C 观察名单（{len(watchlist) if watchlist is not None else 0} 只）\n")
    if watchlist is not None and len(watchlist) > 0:
        lines.append("| 代码 | 名称 | 板块 | 综合评分 |")
        lines.append("|------|------|------|----------|")
        for _, row in watchlist.iterrows():
            lines.append(
                f"| {row['code']} | {row.get('name', row['code'])} | "
                f"{row.get('sector', '-')} | {row.get('composite_score', 0)} |"
            )
    else:
        lines.append("_今日无观察名单_\n")

    # ---- 统计信息 ----
    lines.append(f"\n---\n")
    lines.append(f"**统计**: L2过滤后 {results['l2_filtered_count']} 只 → L4评分 {len(results.get('l4_results', []))} 只")
    lines.append(f" | 反弹引擎 {len(results.get('rebound_picks', []))} 只")
    lines.append(f" | 耗时 {results['elapsed_seconds']}s\n")

    return "\n".join(lines)


def save_brief(content: str, config: dict) -> Path:
    """保存简报到文件（按日期分目录存档）。"""
    out_cfg = config["output"]
    brief_dir = Path(out_cfg["brief_dir"])
    filename = out_cfg["brief_filename"]

    if out_cfg.get("date_based_archive", True):
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = brief_dir / today
    else:
        save_dir = brief_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    save_path.write_text(content, encoding="utf-8")
    logger.info(f"简报已保存: {save_path}")
    return save_path


def main():
    """主入口：运行引擎 → 生成简报 → 保存文件。"""
    parser = argparse.ArgumentParser(description="盘前/盘后选股简报生成器")
    parser.add_argument("--session", default="pre_market",
                        choices=["pre_market", "post_market"],
                        help="选股时段 (默认 pre_market)")
    args, _ = parser.parse_known_args()

    setup_protection()

    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_file=config.get("logging", {}).get("file", "data_cache/engine.log"),
        level=config.get("logging", {}).get("level", "INFO"),
        console=config.get("logging", {}).get("console", True),
    )

    session_label = "盘前" if args.session == "pre_market" else "盘后"

    try:
        logger.info("=" * 60)
        logger.info(f"{session_label}选股简报生成器启动")
        logger.info("=" * 60)

        engine = MultiFactorEngine(config_dict=config)
        results = engine.run(session_type=args.session)

        if "error" in results:
            logger.error(f"引擎运行失败: {results['error']}")
            print(f"ERROR: {results['error']}")
            sys.exit(1)

        brief_content = generate_brief(results, config)
        brief_path = save_brief(brief_content, config)

        print(f"\n{'='*60}")
        print(f"盘前选股简报已生成: {brief_path}")
        print(f"市场环境: {results['regime']['regime']} (仓位上限 {results['regime']['position_cap']:.0%})")
        for cat_name, cat_df in results["categories"].items():
            count = len(cat_df) if cat_df is not None else 0
            print(f"  {cat_name}: {count} 只")
        print(f"耗时: {results['elapsed_seconds']}s")
        print(f"{'='*60}")

        return str(brief_path)

    except KeyboardInterrupt:
        logger.warning("用户中断 (Ctrl+C)，正在清理...")
        return None

    finally:
        teardown_protection()


if __name__ == "__main__":
    main()
