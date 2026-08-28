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
from risk_module import allocate_basket

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

    # 篮子仓位分配配置
    brief_cfg = config.get("brief", {})
    sleeve_weights = brief_cfg.get(
        "sleeve_weights", {"quality": 0.50, "short_term": 0.20, "etf": 0.30})
    alloc_method = brief_cfg.get("method", "score_weighted")
    max_single = brief_cfg.get("max_single_position", 0.08)

    def _fmt_name(row, code):
        """安全获取名称，空/nan 时用代码代替。"""
        n = row.get("name", "")
        if n is None or str(n).lower() in ("nan", "", "none"):
            return code
        return n

    def _etf_settlement(code: str) -> str:
        """A股 ETF 结算方式：货币(511)/债券(511)/黄金(518)/跨境(513) 等为 T+0，
        其余宽基/行业/主题等境内股票型ETF 均为 T+1。此前简报一刀切标 T+0 属错误。"""
        c = str(code).zfill(6)
        if c.startswith(("511", "518", "513")):
            return "T+0"
        return "T+1"

    def _codes_block(codes, per_line: int = 5) -> str:
        """生成可直接复制到同花顺自选的代码块。每行最多 per_line 个（6位代码空格分隔）。

        入参 codes 可为本节 DataFrame（自动安全提取 'code' 列）或代码列表。
        返回 blockquote 形式的多行文本（每行一个 '> '），无代码时返回空串。
        """
        # DataFrame -> 安全提取 code 列 (空 df 或缺失列返回空)
        if isinstance(codes, pd.DataFrame):
            if len(codes) == 0 or "code" not in codes.columns:
                return ""
            codes = codes["code"].tolist()
        norm = [str(c).zfill(6) for c in (codes or []) if c]
        if not norm:
            return ""
        out = []
        for i in range(0, len(norm), per_line):
            out.append("> 📋 **同花顺自选(复制)**: " + " ".join(norm[i:i + per_line]))
        return "\n".join(out)

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
        """根据因子判断建议持仓周期。优先采用 lvrev 内核的 entry_ok 闸门。"""
        roe = row.get("roe", 0)
        mom20 = row.get("momentum_20d", 0)
        score = row.get("composite_score", 50)
        entry_ok = row.get("entry_ok", False)
        # lvrev 内核 entry_ok(趋势向上 MA20>MA60 + 低波 + 不接飞刀 + 底部超跌)
        # 或 高ROE质量票 → 中长线(质量 drift, 已验证 T+2 +2.86%)
        if entry_ok or (pd.notna(roe) and roe > 10 and score > 70):
            return "中长线(5-20日)"
        elif pd.notna(mom20) and abs(mom20) < 5 and score > 65:
            return "中长线(5-15日)"
        else:
            return "短线(1-5日)"

    # 提前初始化所有需要在前置摘要中使用的变量
    quality = categories.get("②A_质量榜")
    if quality is not None and len(quality) > 0:
        long_term = quality[quality.apply(_hold_period, axis=1).str.contains("中长线")].head(8)
    else:
        long_term = pd.DataFrame()
    short_df = categories.get("②B_短线榜")
    if short_df is None:
        short_df = pd.DataFrame()
    etf_picks = results.get("etf_picks", pd.DataFrame())
    watchlist = categories.get("③C_观察名单")

    lines = []
    lines.append(f"# 盘前选股简报 — {today}\n")
    lines.append(f"> 生成时间: {results['timestamp']} | 耗时: {results['elapsed_seconds']}s\n")

    # === 执行摘要（vnpy 风格通知） ===
    mid = len(long_term)
    short = len(short_df)
    etf = len(etf_picks)
    watch = len(watchlist) if watchlist is not None else 0
    regime_name = regime.get("regime", "未知") if isinstance(regime, dict) else str(regime)
    pos_cap_val = regime.get("position_cap", 0.5) if isinstance(regime, dict) else 0.5

    # 板块集中度警告
    top_sectors = long_term["sector"].value_counts().head(3) if len(long_term) > 0 and "sector" in long_term.columns else pd.Series(dtype=int)
    sector_warnings = ""
    for s, cnt in top_sectors.items():
        if cnt >= 3:
            sector_warnings += f"{s}板块集中({cnt}只) "

    verdict = "持仓观望"
    if "多头" in regime_name:
        verdict = "可适度加仓"
    elif "空头" in regime_name:
        verdict = "减仓防御"
    else:
        verdict = "轻仓试探"

    lines.append("> 📋 **执行摘要**\n")
    lines.append(f"> 市场: **{regime_name}**({pos_cap_val:.0%}仓位) | "
                 f"中长线 {mid}只 + 短线 {short}只 + ETF {etf}只 | "
                 f"建议: **{verdict}**\n")
    if sector_warnings:
        lines.append(f"> ⚠️ {sector_warnings}\n")

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
    lines.append("|------|------|------|------|------|----------|")
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

    # 闲资管理提示(用户实操: 空仓/低仓位期用国债逆回购管理待命现金, 约1.5-2%/年零波动; 不做期货故不靠对冲压回撤)
    if pos_cap < 0.5:
        lines.append(f"> 💡 **闲资管理**: 当前仓位上限仅 {pos_cap:.0%}, 大部分资金待命。\n"
                     f"> 维持 **国债逆回购（GC001/GC007 等, 年化~1.5-2%, 零波动、T+0 灵活）** 管理待命现金即可, 别躺零息现金。\n"
                     f"> 若想略提 carry 也可叠加货币ETF（511880/511990, 近乎零波动）, 但逆回购已等效且更灵活。\n"
                     f"> 待 L0 闸门放开、出现达标候选时再赎回买入。\n")

    # ============================================================
    # 二、中长线组合（5-20日持仓）
    # ============================================================
    # 篮子分配: 中长线篮子预算 = 仓位上限 × quality 权重, 按评分加权到各票
    quality_scores = long_term["composite_score"].tolist() if len(long_term) > 0 else []
    quality_codes = long_term["code"].tolist() if len(long_term) > 0 else []
    quality_budget = pos_cap * sleeve_weights.get("quality", 0.50)
    quality_alloc = allocate_basket(quality_scores, quality_budget,
                                    method=alloc_method, max_single=max_single)
    quality_pos_map = {c: a for c, a in zip(quality_codes, quality_alloc)}

    lines.append(f"\n## 二、中长线组合（{len(long_term)} 只，建议持仓 5-20 日）\n")
    if len(long_term) > 0:
        lines.append("| 代码 | 名称 | 股价 | 一手价 | 止损价 | 信号 | 技术面 | 基本面 | 评分 |仓位%| 流动性 | 持有期 | 预期收益 |")
        lines.append("|------|------|------|------|------|------|--------|--------|------|------|------|--------|-------|")
        for _, row in long_term.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            sector = row.get("sector", "-")
            concept = row.get("concept_name", "")
            if not concept or str(concept).lower() in ("nan", "none"):
                concept = "-"
            close = row.get("close", 0)
            close_str = f"{close:.2f}" if pd.notna(close) and close > 0 else "-"
            lot_price = close * 100 if pd.notna(close) and close > 0 else 0
            lot_str = f"{lot_price:.0f}" if lot_price > 0 else "-"
            stop = row.get("stop_loss", 0)
            stop_str = f"{stop:.2f}" if stop > 0 else "-"
            liq = row.get("liquidity_tag", "-")
            score = row.get("composite_score", 0)
            # 信号: 基于评分和 RSI 判定
            rsi6 = row.get("rsi_6")
            if rsi6 is not None and rsi6 < 35:
                signal = f"🔥超卖信号(RSI{rsi6:.0f})"
            elif score >= 80:
                signal = "🔥质量优选"
            elif score >= 65:
                signal = "质量入围"
            else:
                signal = "-"
            # 技术面: MA + MACD（enrich 实际产出 tech_ma/tech_macd/tech_signal；
            # 旧列名 ma_status/macd_signal 已不再生成, 作向后兼容回退, 否则恒为 '-'）
            ma_status = row.get("ma_status") or row.get("tech_ma", "")
            macd_signal = row.get("macd_signal") or row.get("tech_macd", "")
            tech_parts = []
            if ma_status:
                tech_parts.append(str(ma_status))
            if macd_signal:
                tech_parts.append(str(macd_signal))
            tech = "/".join(tech_parts) if tech_parts else "-"
            # 基本面: ROE + 营收增速
            roe_v = row.get("roe")
            rev_g = row.get("revenue_growth")
            fund_parts = []
            if roe_v is not None and pd.notna(roe_v) and roe_v != 0:
                fund_parts.append(f"ROE{roe_v:.1f}%")
            if rev_g is not None and pd.notna(rev_g) and rev_g != 0:
                fund_parts.append(f"营收{rev_g:+.1f}%")
            fund = "/".join(fund_parts) if fund_parts else "-"
            # 篮子仓位: 该票占「总资金」比例 (来自中长线篮子分配)
            pos_ratio = round(quality_pos_map.get(code, 0) * 100, 2)
            period = _hold_period(row)
            net_ret = _net_return(score)
            # 获利概率: 基于因子质量的粗略估计
            factor_count = sum(1 for f in ["roe", "gross_margin", "revenue_growth"]
                              if pd.notna(row.get(f)) and row.get(f) != 0)
            prob = min(85, 50 + factor_count * 8 + max(0, (score - 60) * 0.5))
            lines.append(
                f"| {code} | {name} | {close_str} | {lot_str} | {stop_str} | {signal} | {tech} | {fund} | {score:.1f} | {pos_ratio:.1f}% | {liq} | {period} | {net_ret:+.1f}% |"
            )
    else:
        lines.append("_当前环境不适合中长线持仓_\n")
    # 同花顺复制块：表格结束后只插一次，勿放进行循环（否则 blockquote 断表格）
    _cb = _codes_block(long_term)
    if _cb:
        lines.append("")
        lines.append(_cb)

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

    # 篮子分配: 短线篮子预算 = 仓位上限 × short_term 权重, 按评分加权到各票
    short_scores = short_df["composite_score"].tolist() if len(short_df) > 0 else []
    short_codes = short_df["code"].tolist() if len(short_df) > 0 else []
    short_budget = pos_cap * sleeve_weights.get("short_term", 0.30)
    short_alloc = allocate_basket(short_scores, short_budget,
                                  method=alloc_method, max_single=max_single)
    short_pos_map = {c: a for c, a in zip(short_codes, short_alloc)}

    lines.append(f"\n## 三、短线组合（{len(short_df)} 只，建议持仓 1-5 日）\n")
    if len(short_df) > 0:
        lines.append("| 代码 | 名称 | 股价 | 一手价 | 止损价 | 信号 | 技术面 | 概念 | 概念涨 | 评分 |仓位%| 流动性 | 动量20日 | 预期收益 |")
        lines.append("|------|------|------|------|------|------|--------|------|--------|------|------|------|---------|-------|")
        for _, row in short_df.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            sector = row.get("sector", "-")
            concept = row.get("concept_name", "")
            if not concept or str(concept).lower() in ("nan", "none"):
                concept = "-"
            close = row.get("close", 0)
            close_str = f"{close:.2f}" if pd.notna(close) and close > 0 else "-"
            lot_price = close * 100 if pd.notna(close) and close > 0 else 0
            lot_str = f"{lot_price:.0f}" if lot_price > 0 else "-"
            stop = row.get("stop_loss", 0)
            stop_str = f"{stop:.2f}" if stop > 0 else "-"
            liq = row.get("liquidity_tag", "-")
            score = row.get("composite_score", 0)
            # 篮子仓位: 该票占「总资金」比例 (来自短线篮子分配)
            pos_ratio = round(short_pos_map.get(code, 0) * 100, 2)
            mom20 = _fmt_pct(row.get("momentum_20d"))
            concept_chg = _fmt_pct(row.get("concept_chg"))
            # 短线需关注反弹信号
            decline = row.get("decline_10d", None)
            vol_ratio = row.get("volume_ratio", None)
            rsi6 = row.get("rsi_6")
            if rsi6 is not None and rsi6 < 30:
                signal = f"🔥超跌反弹(RSI{rsi6:.0f})"
            elif decline is not None and vol_ratio is not None:
                signal = f"超跌反弹(量比{vol_ratio})"
            else:
                signal = "-"
            # 技术面
            ma_status = row.get("ma_status") or row.get("tech_ma", "")
            macd_signal = row.get("macd_signal") or row.get("tech_macd", "")
            tech_parts = []
            if ma_status:
                tech_parts.append(str(ma_status))
            if macd_signal:
                tech_parts.append(str(macd_signal))
            tech = "/".join(tech_parts) if tech_parts else "-"
            net_ret = _net_return(score)
            lines.append(
                f"| {code} | {name} | {close_str} | {lot_str} | {stop_str} | {signal} | {tech} | {concept} | {concept_chg} | {score:.1f} | {pos_ratio:.1f}% | {liq} | {mom20} | {net_ret:+.1f}% |"
            )
    else:
        lines.append("_今日无短线候选_\n")
    _cb = _codes_block(short_df)
    if _cb:
        lines.append("")
        lines.append(_cb)

    # 篮子仓位说明 (修复: 单只不再按 position_cap 满算)
    lines.append(
        f"\n> 🧺 **篮子仓位说明**: 表中「仓位%」= 该票占**总资金**的比例, 不是单票满仓。"
        f"环境上限 {pos_cap:.0%} 切分为 中长线×{sleeve_weights.get('quality',0.5):.0%} "
        f"+ 短线×{sleeve_weights.get('short_term',0.3):.0%} + ETF×{sleeve_weights.get('etf',0.2):.0%} "
        f"(权重和=1, 合计=上限)。同一篮子内按评分加权分配, 单只≤总资金{max_single:.0%}。"
        f"本表短线 {len(short_df)} 只合计 ≈ {short_budget:.1%}（占总资金）, "
        f"中长线 {len(long_term)} 只合计 ≈ {quality_budget:.1%}。"
    )

    # ============================================================
    # 四、ETF 组合
    # ============================================================
    lines.append(f"\n## 四、ETF 组合（{len(etf_picks)} 只，A股股票型ETF为**T+1**结算，仅货币/债券/黄金/跨境ETF为T+0）\n")
    if len(etf_picks) > 0:
        lines.append("| 代码 | 名称 | 类型 | 结算 | 动量20日 | 成交额(亿) | 建议 |")
        lines.append("|------|------|------|------|---------|-----------|------|")
        for _, row in etf_picks.iterrows():
            code = row.get("code", "")
            name = _fmt_name(row, code)
            etype = row.get("etf_type", "-")
            settle = _etf_settlement(code)
            mom20 = _fmt_pct(row.get("momentum_20d"))
            amt = f"{row.get('amount', 0)/1e8:.1f}" if row.get("amount") else "-"
            advice = "定投" if row.get("score", 0) > 70 else "关注"
            lines.append(f"| {code} | {name} | {etype} | {settle} | {mom20} | {amt} | {advice} |")
    else:
        lines.append("_ETF 数据源暂不可用_\n")
    _cb = _codes_block(etf_picks)
    if _cb:
        lines.append("")
        lines.append(_cb)

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
    lines.append(f"\n## 六、观察名单（{len(watchlist) if watchlist is not None else 0} 只）\n")
    if watchlist is not None and len(watchlist) > 0:
        lines.append("| 代码 | 名称 | 板块 | 概念 | 评分 | 关注理由 |")
        lines.append("|------|------|------|------|------|----------|")
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
    _cb = _codes_block(watchlist)
    if _cb:
        lines.append("")
        lines.append(_cb)

    # ============================================================
    #  七、持仓追踪与建议
    # ============================================================
    lines.append("\n## 七、持仓追踪与建议\n")
    holdings = config.get("account", {}).get("holdings", {})
    total_assets = config.get("account", {}).get("total_assets", 0)
    pos_cap = results["regime"].get("position_cap", 0.5) if isinstance(results["regime"], dict) else 0.5
    cash = config.get("account", {}).get("available_cash", 0)

    if holdings:
        all_results = (results.get("l4_results", pd.DataFrame()) if isinstance(results.get("l4_results"), pd.DataFrame) else pd.DataFrame())
        cur_price_map = {}
        if len(all_results) > 0 and "code" in all_results.columns:
            for _, r in all_results.iterrows():
                if "close" in all_results.columns:
                    cur_price_map[str(r["code"]).zfill(6)] = r["close"]

        # 持仓实时价优先用 main() 注入的 live 价(results["holding_prices"])。
        # 旧逻辑: 持仓代码不在 l4_results 时直接回退 cost_price 当"当日股价",
        # 导致持仓价=成本、盈亏恒为+0.0%、市值失真(2026-08-25 复盘发现)。
        holding_prices = results.get("holding_prices", {}) or {}

        lines.append("| 代码 | 名称 | 当日股价 | 成本 | 盈亏 | 持仓数 | 市值 | 建议 |")
        lines.append("|------|------|---------|------|------|--------|------|------|")
        total_mv = 0
        for code, info in holdings.items():
            code_str = str(code).zfill(6)
            shares = info.get("shares", 0)
            cost = info.get("cost_price", 0)
            name = info.get("name", code_str)
            # 优先 live 实时价; 缺失再回退候选价/成本(最后手段)
            cur_price = holding_prices.get(code_str)
            if cur_price is None:
                cur_price = cur_price_map.get(code_str, cost)
            pnl = (cur_price - cost) * shares
            pnl_pct = ((cur_price / cost) - 1) * 100 if cost > 0 else 0
            mv = cur_price * shares
            total_mv += mv
            if pnl_pct > 5:
                advice = "✅ 持有"
            elif pnl_pct > -3:
                advice = "🟢 持平"
            else:
                advice = "⚠️ 关注"
            lines.append(
                f"| {code_str} | {name} | {cur_price:.3f} | {cost:.3f} | "
                f"{pnl:+.1f}({pnl_pct:+.1f}%) | {shares} | {mv:.0f} | {advice} |"
            )

        _cb = _codes_block(list(holdings.keys()))
        if _cb:
            lines.append("")
            lines.append(_cb)

        position_pct = (total_mv / total_assets * 100) if total_assets > 0 else 0
        lines.append(f"\n**当前状态**: 总持仓 {total_mv:.0f} 元 | 仓位 {position_pct:.1f}% | 可用资金 {cash:.0f} 元")
        target_pos = pos_cap * 100
        if position_pct < target_pos - 5:
            lines.append(f"**操作建议**: 仓位低于市场允许上限({target_pos:.0f}%)可加仓")
        elif position_pct > target_pos + 5:
            lines.append(f"**操作建议**: 仓位高于市场允许上限({target_pos:.0f}%)应减仓")
        else:
            lines.append(f"**操作建议**: 仓位匹配市场上限({target_pos:.0f}%)，保持")
    else:
        lines.append("_暂无持仓配置_\n")

        # ============================================================
    # 八、统计
    # ============================================================
    lines.append(f"\n---\n")
    lines.append(f"**统计**: L2过滤后 {results.get('l2_filtered_count', 0)} 只 → "
                 f"L4评分 {len(results.get('l4_results', []))} 只")
    lines.append(f" | ETF {len(etf_picks)} 只")
    lines.append(f" | 反弹引擎 {len(results.get('rebound_picks', []))} 只")
    lines.append(f" | 耗时 {results.get('elapsed_seconds', 0)}s\n")

    return "\n".join(lines)


def save_brief(content: str, config: dict) -> Path:
    """保存简报到文件（按日期分目录存档，带分钟时间戳避免同日多次运行覆盖）。

    双写策略：
      - 归档文件：<基名>_<HHMM>.<ext>（每次运行唯一，测试阶段多次运行不丢失）
      - 指针文件：<基名>.<ext>（固定名，永远=最新一次，供 verify_picks / 日常查看）
    """
    out_cfg = config["output"]
    brief_dir = Path(out_cfg["brief_dir"])
    filename = out_cfg["brief_filename"]
    base, ext = Path(filename).stem, Path(filename).suffix

    if out_cfg.get("date_based_archive", True):
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = brief_dir / today
    else:
        save_dir = brief_dir

    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 带分钟时间戳的归档文件（唯一，不会被覆盖）
    ts = datetime.now().strftime("%H%M")
    ts_path = save_dir / f"{base}_{ts}{ext}"
    ts_path.write_text(content, encoding="utf-8")
    logger.info(f"简报已归档(带时间戳): {ts_path}")

    # 2) 固定名指针文件（始终=最新一次，供下游/日常查看；会被覆盖，属预期）
    pointer_path = save_dir / filename
    pointer_path.write_text(content, encoding="utf-8")

    return ts_path


def _premarket_window_ok(config: dict) -> tuple[bool, str]:
    """盘前护栏：校验当前是否仍在合法盘前时段。

    A股 9:30 连续竞价开盘；盘前简报应在开盘前用'前收盘价'生成。
    若任务延迟到开盘后(默认09:30, 北京时)才跑，数据源会返回盘中实时价，
    此时仍冠以'盘前'名号并给买点，等于在下跌途中诱导买入——必须拦截。
    返回 (是否放行, 说明文本)。
    """
    guard = config.get("brief_guard") or {}
    cutoff_str = str(guard.get("premarket_cutoff", "09:30"))
    try:
        h, m = map(int, cutoff_str.split(":"))
        cutoff_min = h * 60 + m
    except Exception:
        cutoff_min = 9 * 60 + 30
    now = datetime.now()  # 机器时区须为 Asia/Shanghai（成都/北京）
    now_min = now.hour * 60 + now.minute
    if now_min < cutoff_min:
        return True, f"当前 {now.hour:02d}:{now.minute:02d} 处于盘前窗口(截止 {cutoff_str})"
    return False, f"当前 {now.hour:02d}:{now.minute:02d} 已越过盘前截止 {cutoff_str}"


def _write_guard_blocked_note(config: dict, detail: str) -> Path:
    """护栏拦截时落一个可见说明文件，避免'无简报=静默'被误读为'8:30正常'。"""
    out_cfg = config.get("output", {})
    brief_dir = Path(out_cfg.get("brief_dir", "history"))
    today = datetime.now().strftime("%Y-%m-%d")
    save_dir = brief_dir / today
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M")
    note = (
        f"# 盘前护栏拦截 ({today} {ts})\n\n"
        f"**未生成盘前选股简报。**\n\n"
        f"原因：{detail}。\n\n"
        f"盘前任务应在开盘前(默认截止 09:30)用前收盘价生成；此时已超过时点，"
        f"数据源将返回盘中实时价，若仍以'盘前'名义给出买点，会在下跌途中诱导买入。\n\n"
        f"## 处理建议\n"
        f"1. 检查机器 8:30 是否因休眠/关机未触发本任务（排程本身正确）。\n"
        f"2. 若确需盘中参考，请改用盘中/盘后流程：\n"
        f"   - `python -m src.daily_brief --session post_market`（盘后）\n"
        f"   - `python -m src.afternoon_review`（15:30 复盘）\n"
        f"3. 本文件仅为拦截记录，不含任何买卖建议。\n"
    )
    path = save_dir / f"盘前护栏拦截_{ts}.md"
    path.write_text(note, encoding="utf-8")
    return path


def main():
    """主入口：运行引擎 → 生成简报 → 保存文件。"""
    parser = argparse.ArgumentParser(description="盘前/盘后选股简报生成器")
    parser.add_argument("--session", default="pre_market",
                        choices=["pre_market", "post_market"],
                        help="选股时段 (默认 pre_market)")
    args, _ = parser.parse_known_args()

    setup_protection()

    config_path = Path(__file__).parent.parent / "config.yaml"
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

        # 盘前护栏：开盘后(默认09:30)禁止用盘中实时价生成'盘前'简报
        if args.session == "pre_market":
            ok, detail = _premarket_window_ok(config)
            if not ok:
                note_path = _write_guard_blocked_note(config, detail)
                msg = (f"[盘前护栏拦截] {detail}。已拒绝以盘中实时价生成'盘前'简报；"
                       f"拦截说明已写入 {note_path}")
                logger.error(msg)
                print("BLOCKED: " + msg)
                sys.exit(2)
            logger.info(detail)

        engine = MultiFactorEngine(config_dict=config)

        # 早盘自包含刷新 ETF 日线（修复 ETF 推荐长期冻结），
        # 不依赖前一日下午复盘的写入。
        try:
            from multifactor import refresh_etf_daily_prices
            etf_res = refresh_etf_daily_prices(engine.price_loader, days=65)
            if etf_res.get("failed"):
                logger.warning(f"ETF 日线刷新存在失败项: {etf_res['failed']}")
        except Exception as e:
            logger.error(f"ETF 日线刷新异常: {e}", exc_info=True)

        results = engine.run(session_type=args.session)

        if "error" in results:
            logger.error(f"引擎运行失败: {results['error']}")
            print(f"ERROR: {results['error']}")
            sys.exit(1)

        # 持仓追踪: 单独拉取持仓代码实时价并注入 results,
        # 避免持仓 ETF 不在 l4_results 时把 cost_price 当"当日股价"(旧 bug)。
        try:
            _hold_cfg = config.get("account", {}).get("holdings", {}) or {}
            if _hold_cfg:
                from westock_helpers import batch_quotes
                _hcodes = [str(c).zfill(6) for c in _hold_cfg.keys()]
                _qmap = batch_quotes(_hcodes)
                _hp = {}
                for _c in _hcodes:
                    _q = _qmap.get(_c)
                    if _q and _q.get("close"):
                        try:
                            _hp[_c] = float(_q["close"])
                        except (TypeError, ValueError):
                            pass
                if _hp:
                    results["holding_prices"] = _hp
                    logger.info(f"持仓实时价已注入: {_hp}")
        except Exception as e:
            logger.warning(f"持仓实时价拉取失败(将回退成本/候选价): {e}")

        brief_content = generate_brief(results, config)
        brief_path = save_brief(brief_content, config)

        # 记录命中追踪（仅追踪"可执行买入候选": ②A质量榜 + ②B短线榜 + ETF）
        # 不再追踪 ③C观察名单(23只弱分观察, 非买入建议)与 ③A持仓/③B卖出建议,
        # 避免每日把大量弱分观察名计入胜率分母、摊薄统计。
        try:
            cats = results.get("categories", {})
            l4 = results.get("l4_results", pd.DataFrame())
            name_map = {}
            for _, r in l4.iterrows():
                n = r.get("name", "")
                if n and str(n).lower() not in ("nan", "none", ""):
                    name_map[str(r["code"]).zfill(6)] = n

            def _fill_name(row):
                nm = row.get("name", "")
                if nm and str(nm).lower() not in ("nan", "none", ""):
                    return nm
                return name_map.get(str(row["code"]).zfill(6), str(row["code"]).zfill(6))

            # 仅追踪可执行买入候选, 各分类分别带 category 标注(供胜率统计精确区分/排除③C)
            for key in ("②A_质量榜", "②B_短线榜"):
                df = cats.get(key)
                if df is not None and len(df) > 0:
                    d = df.copy()
                    d["name"] = d.apply(_fill_name, axis=1)
                    track_picks(d[["code", "name"]], session_type=args.session,
                                category=key)

            # ETF 组合也进入追踪周期(标注 ETF组合)
            etf_picks = results.get("etf_picks", pd.DataFrame())
            if etf_picks is not None and len(etf_picks) > 0 and "code" in etf_picks.columns:
                cols = [c for c in ("code", "name") if c in etf_picks.columns]
                d = etf_picks[cols].copy()
                d["name"] = d.apply(_fill_name, axis=1)
                track_picks(d[["code", "name"]], session_type=args.session,
                            category="ETF组合")
        except Exception as e:
            logger.warning(f"命中追踪记录失败: {e}")

        print(f"\n{'='*60}")
        print(f"盘前选股简报已生成: {brief_path}")

        # 生成 HTML 版本（归档 + 指针双写，与 Markdown 一致）
        try:
            from html_report import generate_html
            html_content = generate_html(results, config)
            # 时间戳归档 HTML（唯一，不覆盖）
            html_ts = Path(str(brief_path).replace(".md", ".html"))
            html_ts.write_text(html_content, encoding="utf-8")
            # 固定名指针 HTML（最新一次）
            pointer_name = Path(config["output"]["brief_filename"]).with_suffix(".html").name
            html_pointer = brief_path.parent / pointer_name
            html_pointer.write_text(html_content, encoding="utf-8")
            print(f"HTML 简报已生成: {html_ts}")
            print(f"HTML 指针已更新: {html_pointer}")
        except Exception as e:
            logger.warning(f"HTML 简报生成失败: {e}")
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
