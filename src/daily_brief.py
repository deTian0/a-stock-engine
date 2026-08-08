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
from pathlib import Path
from datetime import datetime

import pandas as pd

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from multifactor import MultiFactorEngine

logger = logging.getLogger(__name__)


def setup_logging(config: dict):
    """配置日志。"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "data_cache/engine.log")
    console = log_cfg.get("console", True)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


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
            median = results["l4_results"]["composite_score"].median() if len(results["l4_results"]) > 0 else 50
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
    config_path = "config/config.yaml"
    config_path = Path(__file__).parent.parent / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger.info("=" * 60)
    logger.info("盘前选股简报生成器启动")
    logger.info("=" * 60)

    # 运行选股引擎
    engine = MultiFactorEngine(config_path=str(config_path))
    results = engine.run()

    if "error" in results:
        logger.error(f"引擎运行失败: {results['error']}")
        print(f"ERROR: {results['error']}")
        sys.exit(1)

    # 生成简报
    brief_content = generate_brief(results, config)

    # 保存简报
    brief_path = save_brief(brief_content, config)

    # 输出摘要
    print(f"\n{'='*60}")
    print(f"盘前选股简报已生成: {brief_path}")
    print(f"市场环境: {results['regime']['regime']} (仓位上限 {results['regime']['position_cap']:.0%})")
    for cat_name, cat_df in results["categories"].items():
        count = len(cat_df) if cat_df is not None else 0
        print(f"  {cat_name}: {count} 只")
    print(f"耗时: {results['elapsed_seconds']}s")
    print(f"{'='*60}")

    return str(brief_path)


if __name__ == "__main__":
    main()
