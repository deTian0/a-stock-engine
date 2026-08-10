"""
afternoon_review.py — 每日盘后复盘（15:30 执行）

复盘内容:
  1. 今日最强板块 Top 5 及涨幅
  2. 每板块最强 3-5 只个股
  3. 上涨动因分析（资金流入、量比、技术形态）
  4. 早盘推荐 T+0 表现验证
  5. 策略优化建议

用法:
    python -m src.afternoon_review
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from westock_cli import get_cli
from database import get_db
from local_price_loader import LocalPriceLoader
from pick_tracker import get_tracking_report, track_picks

logger = logging.getLogger(__name__)


def review_sectors(cli, price_loader) -> str:
    """
    板块复盘: 获取今日最强板块，分析上涨动因。
    """
    lines = []
    lines.append(f"# 盘后复盘报告 — {datetime.now().strftime('%Y-%m-%d')}\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%H:%M')}\n")

    # ============================================================
    # 1. 最强板块
    # ============================================================
    lines.append("## 一、今日最强板块 Top 5\n")
    try:
        sector_list = cli.get_sector_list()
    except Exception:
        sector_list = pd.DataFrame()

    if len(sector_list) > 0:
        # 按涨跌幅排序
        if "change_pct" in sector_list.columns:
            top_sectors = sector_list.nlargest(5, "change_pct")
        else:
            top_sectors = sector_list.head(5)

        lines.append("| 板块 | 涨幅 | 成交额(亿) | 5日涨幅 | 资金趋势 |")
        lines.append("|------|------|-----------|---------|----------|")
        for _, row in top_sectors.iterrows():
            name = row.get("name", row.get("sector_name", "-"))
            chg = f"{row.get('change_pct', 0):.1f}%" if pd.notna(row.get("change_pct")) else "-"
            amt = f"{row.get('amount', 0)/1e8:.1f}" if row.get("amount") else "-"
            chg5 = f"{row.get('change_5d', 0):.1f}%" if pd.notna(row.get("change_5d")) else "-"
            # 资金趋势: 对比成交额变化
            amt_chg_val = row.get("amount_change", 0)
            trend = "📈 放量" if amt_chg_val > 0 else ("📉 缩量" if amt_chg_val < 0 else "→ 持平")
            lines.append(f"| {name} | {chg} | {amt} | {chg5} | {trend} |")
    else:
        lines.append("_板块数据暂不可用_\n")

    # ============================================================
    # 2. 板块内最强个股
    # ============================================================
    lines.append("\n## 二、各板块最强个股\n")
    top_sector_names = []
    if len(sector_list) > 0 and "name" in sector_list.columns:
        top_sector_names = sector_list.nlargest(5, "change_pct")["name"].tolist()

    sector_mapping = {}
    try:
        sector_mapping = cli.get_sector_mapping()
    except Exception:
        pass

    # 获取全市场涨跌幅（用于板块内筛选）
    try:
        all_stocks = cli.get_stock_list()
    except Exception:
        all_stocks = pd.DataFrame()

    for sector_name in top_sector_names[:5]:
        lines.append(f"\n### {sector_name}\n")
        # 找出该板块的股票
        sector_codes = [c for c, s in sector_mapping.items() if s == sector_name]

        if len(sector_codes) > 0 and len(all_stocks) > 0:
            sector_stocks = all_stocks[all_stocks["code"].isin(sector_codes)]
            if len(sector_stocks) > 0 and "change_pct" in sector_stocks.columns:
                top = sector_stocks.nlargest(5, "change_pct")
                lines.append("| 代码 | 名称 | 涨幅 | 量比 | 成交额(亿) | 动因分析 |")
                lines.append("|------|------|------|------|-----------|----------|")
                for _, row in top.iterrows():
                    code = row.get("code", "")
                    name = row.get("name", code)
                    chg = f"{row.get('change_pct', 0):.1f}%" if pd.notna(row.get("change_pct")) else "-"
                    vol_r = f"{row.get('volume_ratio', 0):.2f}" if pd.notna(row.get("volume_ratio")) else "-"
                    amt = f"{row.get('amount', 0)/1e8:.1f}" if pd.notna(row.get("amount", 0)) else "-"
                    # 动因分析
                    causes = _analyze_cause(row, price_loader, code)
                    lines.append(f"| {code} | {name} | {chg} | {vol_r} | {amt} | {', '.join(causes[:2])} |")
            else:
                lines.append("_该板块个股数据不足_\n")
        else:
            lines.append("_板块映射不可用_\n")

    # ============================================================
    # 3. 早盘推荐验证
    # ============================================================
    lines.append("\n## 三、早盘推荐 T+0 验证\n")
    try:
        db = get_db()
        latest = db.get_latest_run()
        if latest:
            lines.append(f"早盘 run_id={latest['run_id']}, {latest['date']}\n")
            detail = db.get_run_detail(latest['run_id'])
            picks = detail.get("picks", [])

            # 按分类汇总
            by_cat = defaultdict(list)
            for p in picks:
                by_cat[p.get("category", "其他")].append(p)

            for cat, cat_picks in by_cat.items():
                lines.append(f"\n### {cat}\n")
                if len(cat_picks) > 0:
                    lines.append("| 代码 | 名称 | 选股评分 | T+0表现 | 建议 |")
                    lines.append("|------|------|----------|---------|------|")
                    # 尝试获取今日涨跌幅
                    for p in cat_picks[:10]:
                        code = p.get("code", "")
                        name = p.get("name", code)
                        score = p.get("composite_score", 0)

                        # 从实时数据中获取今日表现
                        t0_perf = "-"
                        advice = "-"
                        if len(all_stocks) > 0 and "code" in all_stocks.columns:
                            match = all_stocks[all_stocks["code"] == code]
                            if len(match) > 0 and "change_pct" in match.columns:
                                today_chg = match.iloc[0].get("change_pct", 0)
                                t0_perf = f"{today_chg:+.1f}%" if pd.notna(today_chg) else "-"
                                if pd.notna(today_chg):
                                    advice = "✅ 符合预期" if today_chg > 0 else "⚠️ 暂时偏弱"
                        lines.append(f"| {code} | {name} | {score:.0f} | {t0_perf} | {advice} |")
        else:
            lines.append("_暂无早盘选股记录_\n")
    except Exception as e:
        logger.warning(f"T+0验证失败: {e}")
        lines.append(f"_验证数据获取失败: {e}_\n")

    # ============================================================
    # 4. 策略优化建议
    # ============================================================
    lines.append("\n## 四、策略优化建议\n")
    try:
        db = get_db()
        t2_stats = db.get_t2_stats(days=30)
        if t2_stats.get("count", 0) > 0:
            lines.append(f"- 近30日 T+2 胜率: **{t2_stats['win_rate']}%** ({t2_stats['positive']}/{t2_stats['count']})")
            lines.append(f"- 平均收益率: **{t2_stats['avg_return']}%**, 中位数: {t2_stats['median_return']}%")
            lines.append(f"- 最大收益: {t2_stats['max_return']}%, 最小: {t2_stats['min_return']}%")
            if t2_stats['win_rate'] < 45:
                lines.append("- ⚠️ 胜率偏低，建议收紧因子阈值或增加基本面权重")
            elif t2_stats['win_rate'] > 60:
                lines.append("- ✅ 策略表现良好，可考虑适度加仓")
    except Exception as e:
        logger.warning(f"策略分析失败: {e}")

    # 因子有效性
    try:
        db = get_db()
        corrs = db.get_factor_effectiveness(days=60)
        if corrs:
            lines.append("\n### 因子有效性（近60日与T+2收益的相关性）\n")
            valid = {k: v for k, v in corrs.items() if v is not None}
            sorted_corrs = sorted(valid.items(), key=lambda x: abs(x[1]), reverse=True)
            lines.append("| 因子 | 相关性 | 建议 |")
            lines.append("|------|--------|------|")
            for factor, corr in sorted_corrs[:8]:
                bar = "🟢" if abs(corr) > 0.1 else ("🟡" if abs(corr) > 0.05 else "⚪")
                suggestion = "权重不变"
                if corr > 0.1:
                    suggestion = "可增加权重"
                elif corr < -0.1:
                    suggestion = "考虑降低权重"
                lines.append(f"| {factor} | {bar} {corr:.3f} | {suggestion} |")
    except Exception as e:
        logger.warning(f"因子分析失败: {e}")

    lines.append(f"\n---\n")
    lines.append(f"*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # ============================================================
    # 5. 命中追踪
    # ============================================================
    try:
        lines.append("\n")
        lines.append(get_tracking_report())
    except Exception as e:
        logger.warning(f"命中追踪报告失败: {e}")

    return "\n".join(lines)


def _analyze_cause(row, price_loader, code: str) -> list[str]:
    """分析个股上涨动因。"""
    causes = []
    # 量比
    vol_ratio = row.get("volume_ratio")
    if pd.notna(vol_ratio) and vol_ratio > 1.5:
        causes.append(f"放量(量比{vol_ratio:.1f})")
    # 振幅
    amp = row.get("amplitude")
    if pd.notna(amp) and amp > 5:
        causes.append(f"高波动(振幅{amp:.1f}%)")
    # 板块联动(由调用方判断)
    # 技术形态
    try:
        df = price_loader.get_price(code, days=20)
        if len(df) >= 20:
            close = df["close"].values
            ma5 = close[-5:].mean()
            # 突破5日新高
            if close[-1] >= close[-20:].max() * 0.98:
                causes.append("突破近期高点")
            # 连续阳线
            if len(close) >= 3 and all(close[-(i+1)] > close[-(i+2)] for i in range(2)):
                causes.append("连阳走势")
    except Exception:
        pass

    if not causes:
        # 检查板块联动
        sector = row.get("sector", "")
        if sector:
            causes.append(f"{sector}板块联动")
        else:
            causes.append("资金推动")

    return causes


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data_cache/engine.log", encoding="utf-8"),
        ]
    )

    logger.info("=" * 60)
    logger.info("盘后复盘生成器启动")
    logger.info("=" * 60)

    cli = get_cli()
    price_loader = LocalPriceLoader()

    try:
        content = review_sectors(cli, price_loader)

        # 保存
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path("briefs") / today
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "盘后复盘报告.md"
        save_path.write_text(content, encoding="utf-8")

        # 记录盘后命中追踪
        try:
            all_stocks = cli.get_stock_list()
            if len(all_stocks) > 0 and "change_pct" in all_stocks.columns:
                top_gainers = all_stocks.nlargest(30, "change_pct")
                if "code" in top_gainers.columns:
                    track_picks(top_gainers, session_type="post_market")
        except Exception as e:
            logger.debug(f"盘后追踪记录跳过: {e}")

        logger.info(f"复盘报告已保存: {save_path}")
        print(f"\n复盘报告: {save_path}")
    except Exception as e:
        logger.error(f"复盘失败: {e}", exc_info=True)
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
