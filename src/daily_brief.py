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

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from multifactor import MultiFactorEngine
from database import get_db
from guard import setup_protection, teardown_protection, setup_logging
from pick_tracker import track_picks

logger = logging.getLogger(__name__)


def generate_brief(results: dict, config: dict) -> str:
    """
    根据引擎结果生成 Markdown 格式的盘前简报。
    包含: 持仓周期分类、磨损调整收益、ETF 选股、获利概率评估。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    regime = results["regime"]
    categories = results["categories"]
    trade_cfg = config.get("trade", {})
    stock_cost = trade_cfg.get("stock_cost", 0.0013)
    etf_cost = trade_cfg.get("etf_cost", 0.00017)

    def _fmt_name(row, code):
        """安全获取名称，空/nan 时用代码代替。"""
        n = row.get("name", "")
        if n is None or str(n).lower() in ("nan", "", "none"):
            return code
        return n

    def _fmt_pct(val, default="-"):
        """安全格式化百分比。"""
        if val is None or not pd.notna(val):
            return default
        return f"{val:.1f}%"
        return f"{val:.1f}%"

    def _net_return(score, cost=stock_cost):
        """预计磨损后收益: 评分 → 粗略预期收益 → 扣除成本。
        评分50=无超额收益, 每1分≈0.15%预期超额收益。"""
        excess = (score - 50) * 0.15  # 超额alpha估算
        return max(excess - cost * 100, -cost * 100)  # 扣磨损, 不低于成本

    def _hold_period(row):
        """根据因子判断建议持仓周期。"""
        roe = row.get("roe", 0)
        mom20 = row.get("momentum_20d", 0)
        score = row.get("composite_score", 50)
        # 高ROE+稳定动量 → 中长线; 纯动量驱动 → 短线
        if pd.notna(roe) and roe > 10 and score > 75:
            return "中长线(5-20日)"
        elif pd.notna(mom20) and abs(mom20) < 5 and score > 70:
            return "中长线(5-15日)"
        else:
            return "短线(1-5日)"

    lines = []
    lines.append(f"# 盘前选股简报 — {today}\n")
    lines.append(f"> 生成时间: {results['timestamp']} | 耗时: {results['elapsed_seconds']}s\n")

    # ============================================================
    # 一、市场环境
    # ============================================================
    lines.append("## 一、市场环境判断\n")
    pos_cap = regime['position_cap'] if isinstance(regime, dict) else 0.5
    judgment = regime.get('judgment', '') if isinstance(regime, dict) else ''
    lines.append(f"**当前环境: {regime.get('regime','未知') if isinstance(regime,dict) else regime}** "
                 f"| 仓位上限: {pos_cap:.0%}\n")
    if judgment:
        lines.append(f"_{judgment}_\n")
    lines.append("\n| 指数 | 收盘 | MA20 | MA60 | 站上MA60 |")
    lines.append("|------|------|------|------|----------|")
    if isinstance(regime, dict):
        for code, info in regime.get("indices", {}).items():
            if "error" in info:
                lines.append(f"| {info.get('name', code)} | 获取失败 | - | - | - |")
            else:
                above = "✅" if info.get("above_ma") else "❌"
                lines.append(
                    f"| {info.get('name', code)} | {info.get('close', '-')} | "
                    f"{info.get('ma_short', '-')} | {info.get('ma_long', '-')} | {above} |"
                )
    lines.append("")

    # ============================================================
    # 二、中长线组合（5-20日持仓）
    # ============================================================
    quality = categories.get("②A_质量榜")
    if quality is not None and len(quality) > 0:
        long_term = quality[quality.apply(_hold_period, axis=1).str.contains("中长线")].head(8)
    else:
        long_term = pd.DataFrame()

    lines.append(f"\n## 二、中长线组合（{len(long_term)} 只，建议持仓 5-20 日）\n")
    if len(long_term) > 0:
        lines.append("| 代码 | 名称 | 板块 | 评分 | 持有期 | 预期净收益 | 获利概率 |")
        lines.append("|------|------|------|------|--------|-----------|----------|")
        for _, row in long_term.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            sector = row.get("sector", "-")
            score = row.get("composite_score", 0)
            period = _hold_period(row)
            net_ret = _net_return(score)
            # 获利概率: 基于因子质量的粗略估计
            factor_count = sum(1 for f in ["roe", "gross_margin", "revenue_growth"]
                              if pd.notna(row.get(f)) and row.get(f) != 0)
            prob = min(85, 50 + factor_count * 8 + max(0, (score - 60) * 0.5))
            lines.append(
                f"| {code} | {name} | {sector} | {score:.1f} | {period} | "
                f"{net_ret:+.1f}% | {prob:.0f}% |"
            )
    else:
        lines.append("_当前环境不适合中长线持仓_\n")

    # ============================================================
    # 三、短线组合（1-5日持仓）
    # ============================================================
    short_df = categories.get("②B_短线榜")
    if short_df is not None and len(short_df) > 0:
        # 如果有质量榜的短线部分也合并进来
        if quality is not None and len(quality) > 0:
            short_quality = quality[quality.apply(_hold_period, axis=1).str.contains("短线")]
            short_df = pd.concat([short_df, short_quality], ignore_index=True).drop_duplicates(subset=["code"]).head(8)
    else:
        short_df = pd.DataFrame()

    lines.append(f"\n## 三、短线组合（{len(short_df)} 只，建议持仓 1-5 日）\n")
    if len(short_df) > 0:
        lines.append("| 代码 | 名称 | 板块 | 评分 | 动量20日 | 量比信号 | 预期净收益 |")
        lines.append("|------|------|------|------|---------|---------|-----------|")
        for _, row in short_df.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            sector = row.get("sector", "-")
            score = row.get("composite_score", 0)
            mom20 = _fmt_pct(row.get("momentum_20d"))
            # 短线需关注反弹信号
            decline = row.get("decline_10d", None)
            vol_ratio = row.get("volume_ratio", None)
            if decline is not None and vol_ratio is not None:
                signal = f"超跌反弹(量比{vol_ratio})"
            else:
                signal = "-"
            net_ret = _net_return(score)
            lines.append(
                f"| {code} | {name} | {sector} | {score:.1f} | {mom20} | "
                f"{signal} | {net_ret:+.1f}% |"
            )
    else:
        lines.append("_今日无短线候选_\n")

    # ============================================================
    # 四、ETF 组合
    # ============================================================
    etf_picks = results.get("etf_picks", pd.DataFrame())
    lines.append(f"\n## 四、ETF 组合（{len(etf_picks)} 只，T+0/低磨损 {etf_cost*100:.2f}%）\n")
    if len(etf_picks) > 0:
        lines.append("| 代码 | 名称 | 类型 | 动量20日 | 成交额(亿) | 建议 |")
        lines.append("|------|------|------|---------|-----------|------|")
        for _, row in etf_picks.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            etype = row.get("etf_type", "-")
            mom20 = _fmt_pct(row.get("momentum_20d"))
            amt = f"{row.get('amount', 0)/1e8:.1f}" if row.get("amount") else "-"
            advice = "定投" if row.get("score", 0) > 70 else "关注"
            lines.append(f"| {code} | {name} | {etype} | {mom20} | {amt} | {advice} |")
    else:
        lines.append("_ETF 数据源暂不可用_\n")

    # ============================================================
    # 五、持仓与操作
    # ============================================================
    holdings = categories.get("③A_持仓")
    lines.append(f"\n## 五、当前持仓与操作建议（{len(holdings) if holdings is not None else 0} 只）\n")
    if holdings is not None and len(holdings) > 0:
        lines.append("| 代码 | 名称 | 评分 | 板块 | 持仓周期建议 | 操作 |")
        lines.append("|------|------|------|------|-------------|------|")
        for _, row in holdings.iterrows():
            score = row.get("composite_score", 0)
            period = _hold_period(row)
            median = results.get("l4_results", pd.DataFrame()).get("composite_score", pd.Series([50])).median() if len(results.get("l4_results", [])) > 0 else 50
            advice = "⚠️ 减仓" if score < median else "✅ 持有"
            lines.append(
                f"| {row.get('code','')} | {row.get('name','')} | {score:.0f} | "
                f"{row.get('sector','-')} | {period} | {advice} |"
            )
    else:
        lines.append("_当前无持仓_\n")

    # ============================================================
    # 六、观察名单
    # ============================================================
    watchlist = categories.get("③C_观察名单")
    lines.append(f"\n## 六、观察名单（{len(watchlist) if watchlist is not None else 0} 只）\n")
    if watchlist is not None and len(watchlist) > 0:
        lines.append("| 代码 | 名称 | 板块 | 评分 | 关注理由 |")
        lines.append("|------|------|------|------|----------|")
        for _, row in watchlist.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            sector = row.get("sector", "-")
            score = row.get("composite_score", 0)
            # 关注理由
            reasons = []
            if pd.notna(row.get("roe")) and row.get("roe", 0) > 10:
                reasons.append(f"高ROE{row.get('roe',0):.0f}%")
            if pd.notna(row.get("momentum_20d")) and row.get("momentum_20d", 0) > 0:
                reasons.append("动量转正")
            if not reasons:
                reasons.append("综合因子")
            lines.append(f"| {code} | {name} | {sector} | {score:.1f} | {', '.join(reasons[:2])} |")
    else:
        lines.append("_今日无观察名单_\n")

    # ============================================================
    # 七、统计
    # ============================================================
    lines.append(f"\n---\n")
    lines.append(f"**统计**: L2过滤后 {results.get('l2_filtered_count', 0)} 只 → "
                 f"L4评分 {len(results.get('l4_results', []))} 只")
    lines.append(f" | ETF {len(etf_picks)} 只")
    lines.append(f" | 反弹引擎 {len(results.get('rebound_picks', []))} 只")
    lines.append(f" | 耗时 {results.get('elapsed_seconds', 0)}s\n")

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

    config_path = Path(__file__).parent.parent / "config/config.yaml"
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

        # 记录命中追踪
        try:
            all_picks = pd.concat(
                [df for df in results["categories"].values() if df is not None and len(df) > 0],
                ignore_index=True
            )
            if len(all_picks) > 0:
                # 补齐名称：从 L4 结果或 stock_list 中查找
                l4 = results.get("l4_results", pd.DataFrame())
                name_map = {}
                for _, r in l4.iterrows():
                    n = r.get("name", "")
                    if n and str(n).lower() not in ("nan", "none", ""):
                        name_map[str(r["code"]).zfill(6)] = n
                # L4没找到的从 stock_list 补齐
                if "name" in all_picks.columns:
                    all_picks["name"] = all_picks.apply(
                        lambda r: name_map.get(str(r["code"]).zfill(6), r["code"]), axis=1
                    )
                track_picks(all_picks, session_type=args.session)
        except Exception as e:
            logger.warning(f"命中追踪记录失败: {e}")

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
