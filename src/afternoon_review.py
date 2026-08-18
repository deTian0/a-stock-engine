"""
afternoon_review.py — 每日盘后复盘（15:30 执行）

复盘内容:
  1. 今日最强板块 Top 5 及涨幅
  2. 每板块最强 3-5 只个股
  3. 上涨动因分析（资金流入、量比、技术形态）
  4. 早盘推荐当日表现验证（个股 T+1，仅展示当日涨跌，非已实现收益）
  5. 策略优化建议

输出:
  history/<日期>/盘后复盘报告.md   （原 Markdown 版，保持兼容）
  history/<日期>/盘后复盘报告.html （新增 HTML 版，ECharts 可视化，便于阅读）

用法:
    python -m src.afternoon_review
"""

import sys
import os
import re
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
from pick_tracker import get_tracking_report, get_tracking_summary, track_picks

logger = logging.getLogger(__name__)

_QUOTE_COLS = ["change_pct", "close", "amount", "volume", "volume_ratio", "amplitude"]


def _inject_westock_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """注入 westock-data 实时行情（tushare/新浪回退列表可能缺 change_pct/close/amount）。

    幂等：仅在 change_pct/close/amount 全缺时才触发，注入一次后再次调用为 no-op。
    """
    if df is None or len(df) == 0 or "code" not in df.columns:
        return df
    need = (
        "change_pct" not in df.columns or df["change_pct"].isna().all()
        or "close" not in df.columns or df["close"].isna().all()
        or "amount" not in df.columns or df["amount"].isna().all()
    )
    if not need:
        return df
    from westock_helpers import batch_quotes
    quote_map = batch_quotes(df["code"].astype(str).str.zfill(6).tolist())
    if not quote_map:
        return df
    qdf = pd.DataFrame.from_dict(quote_map, orient="index").reset_index().rename(columns={"index": "code"})
    qdf["code"] = qdf["code"].astype(str).str.zfill(6)
    qmap = qdf.set_index("code")
    codes_z = df["code"].astype(str).str.zfill(6)
    for col in _QUOTE_COLS:
        if col in qdf.columns and (col not in df.columns or df[col].isna().all()):
            df[col] = codes_z.map(qmap[col])
    logger.info(f"已注入westock实时行情: {len(quote_map)} 只 (涨跌幅覆盖 {df['change_pct'].notna().sum()} 只)")
    return df


def review_sectors(cli, price_loader, all_stocks: pd.DataFrame = None,
                    preloaded_prices: dict = None, config: dict = None) -> tuple[str, dict]:
    """
    板块复盘: 获取今日最强板块，分析上涨动因。

    返回: (markdown_text, structured_data)
      - markdown_text: 与历史版本完全一致的 Markdown 复盘内容（向后兼容）
      - structured_data: 供 generate_review_html() 渲染 HTML 的结构化字典
    """
    if config is None:
        config = {}
    if all_stocks is None:
        all_stocks = pd.DataFrame()
    if preloaded_prices is None:
        preloaded_prices = {}

    # 结构化数据容器（供 HTML 渲染）
    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%H:%M"),
        "top_sectors": [],        # 最强板块 Top5: {sector, avg_chg, stock_count, max_chg, amount_yi}
        "sector_stocks": {},      # 板块名 -> [个股 dict]
        "t0_verify": None,        # 早盘 T+0 验证: {run_id, date, by_cat}
        "hit_stats": None,        # 命中统计: {total_stocks,hits,cycles,active,multi_hit,...}
        "tracking_md": "",        # 命中追踪原始 Markdown
    }

    lines = []
    lines.append(f"# 盘后复盘报告 — {data['date']}\n")
    lines.append(f"> 生成时间: {data['generated_at']}\n")

    # ============================================================
    # 1. 最强板块 — 用股票列表按行业聚合
    # ============================================================
    lines.append("## 一、今日最强板块 Top 5\n")

    # 使用传入的 all_stocks（避免重复 CLI 调用）
    stock_list = all_stocks if len(all_stocks) > 0 else pd.DataFrame()

    sector_mapping = {}
    sector_stocks = {}
    try:
        codes_for_mapping = stock_list["code"].astype(str).str.zfill(6).unique().tolist() if len(stock_list) > 0 and "code" in stock_list.columns else []
        sector_mapping = cli.get_sector_mapping(codes_for_mapping[:500] if codes_for_mapping else None)
    except Exception:
        pass

    # 按行业聚合: 平均涨幅、总成交额
    if len(stock_list) > 0 and "code" in stock_list.columns:
        # 排除北交所
        codes = stock_list["code"].astype(str).str.zfill(6)
        stock_list = stock_list[codes.str.startswith(("0","1","3","5","6"))]
        # 注入 westock-data 实时行情（tushare/新浪回退列表可能缺 change_pct/close/amount）
        stock_list = _inject_westock_quotes(stock_list)
        stock_list["sector"] = stock_list["code"].astype(str).str.zfill(6).map(sector_mapping).fillna("综合")
        agg_spec = {
            "avg_chg": ("change_pct", "mean"),
            "stock_count": ("code", "count"),
            "max_chg": ("change_pct", "max"),
        }
        if "amount" in stock_list.columns:
            agg_spec["total_amt"] = ("amount", "sum")
        sector_agg = stock_list.groupby("sector").agg(**agg_spec).reset_index()
        sector_agg = sector_agg[sector_agg["stock_count"] >= 3]  # 至少3只股票
        top_sectors = sector_agg.nlargest(5, "avg_chg")

        lines.append("| 板块 | 平均涨幅 | 股票数 | 龙头涨幅 | 成交额(亿) |")
        lines.append("|------|---------|--------|---------|-----------|")
        for _, row in top_sectors.iterrows():
            name = row["sector"]
            avg = f"{row['avg_chg']:.1f}%" if pd.notna(row["avg_chg"]) else "-"
            cnt = int(row["stock_count"])
            top = f"{row['max_chg']:.1f}%" if pd.notna(row["max_chg"]) else "-"
            amt = f"{row['total_amt']/1e8:.0f}" if pd.notna(row.get("total_amt")) else "-"
            lines.append(f"| {name} | {avg} | {cnt} | {top} | {amt} |")
            data["top_sectors"].append({
                "sector": name,
                "avg_chg": float(row["avg_chg"]) if pd.notna(row["avg_chg"]) else None,
                "stock_count": cnt,
                "max_chg": float(row["max_chg"]) if pd.notna(row["max_chg"]) else None,
                "amount_yi": float(row["total_amt"])/1e8 if pd.notna(row.get("total_amt")) else None,
            })
        top_sector_names = top_sectors["sector"].tolist()
    else:
        lines.append("_板块数据暂不可用_\n")
        top_sector_names = []
        stock_list = pd.DataFrame()

    # ============================================================
    # 2. 板块内最强个股
    # ============================================================
    lines.append("\n## 二、各板块最强个股\n")

    for sector_name in top_sector_names[:5]:
        lines.append(f"\n### {sector_name}\n")
        if "sector" in stock_list.columns:
            sector_df = stock_list[stock_list["sector"] == sector_name]
        else:
            codes = [c for c, s in sector_mapping.items() if s == sector_name]
            sector_df = stock_list[stock_list["code"].isin(codes)] if "code" in stock_list.columns else pd.DataFrame()

        if len(sector_df) > 0 and "change_pct" in sector_df.columns:
            top = sector_df.nlargest(5, "change_pct")
            lines.append("| 代码 | 名称 | 当日股价 | 一手价格 | 止损 | 涨幅 | 量比 | 成交额(亿) | 动因 |")
            lines.append("|------|------|---------|---------|------|------|------|-----------|------|")
            sec_list = []
            for _, row in top.iterrows():
                code = str(row.get("code", ""))
                name = str(row.get("name", code))
                if name.lower() in ("nan", "none", ""):
                    name = code
                # 当日股价 (从 close 列)
                close = row.get("close", 0)
                close_str = f"{close:.2f}" if pd.notna(close) and close > 0 else "-"
                lot_str = f"{close * 100:.0f}" if pd.notna(close) and close > 0 else "-"
                stop = _compute_stop_loss(preloaded_prices.get(code), close)
                stop_str = f"{stop:.2f}" if stop is not None else "-"
                chg = f"{row.get('change_pct', 0):.1f}%" if pd.notna(row.get("change_pct")) else "-"
                vol_r = f"{row.get('volume_ratio', 0):.2f}" if pd.notna(row.get("volume_ratio")) else "-"
                amt = f"{row.get('amount', 0)/1e8:.1f}" if pd.notna(row.get("amount")) else "-"
                causes = _analyze_cause(row, preloaded_prices, code)
                lines.append(f"| {code} | {name} | {close_str} | {lot_str} | {stop_str} | {chg} | {vol_r} | {amt} | {', '.join(causes[:2])} |")
                sec_list.append({
                    "code": code,
                    "name": name if name != code else code,
                    "close": float(close) if pd.notna(close) and close > 0 else None,
                    "lot": int(close * 100) if pd.notna(close) and close > 0 else None,
                    "stop_loss": stop,
                    "chg": float(row.get("change_pct", 0)) if pd.notna(row.get("change_pct")) else None,
                    "vol_ratio": float(row.get("volume_ratio", 0)) if pd.notna(row.get("volume_ratio")) else None,
                    "amount_yi": float(row.get("amount", 0))/1e8 if pd.notna(row.get("amount")) else None,
                    "causes": causes[:2],
                })
            data["sector_stocks"][sector_name] = sec_list
        else:
            lines.append("_个股数据不足_\n")

    # ============================================================
    # 3. 早盘推荐验证
    # ============================================================
    lines.append("\n## 三、早盘推荐当日表现\n\n> ⚠️ 个股为 **T+1** 结算，当日买入不可当日卖出；下表仅展示「当日涨跌」，并非已实现收益。ETF 视类型（货币/债券/黄金/跨境为 T+0，其余 T+1）。\n")
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

            t0_data = {"run_id": latest["run_id"], "date": latest["date"], "by_cat": {}}
            for cat, cat_picks in by_cat.items():
                lines.append(f"\n### {cat}\n")
                if len(cat_picks) > 0:
                    lines.append("| 代码 | 名称 | 选股评分 | 当日表现 | 建议 |")
                    lines.append("|------|------|----------|---------|------|")
                    # 尝试获取今日涨跌幅
                    cat_rows = []
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
                        cat_rows.append({
                            "code": code, "name": name, "score": float(score),
                            "t0": t0_perf, "advice": advice,
                        })
                    t0_data["by_cat"][cat] = cat_rows
            data["t0_verify"] = t0_data
        else:
            lines.append("_暂无早盘选股记录_\n")
    except Exception as e:
        logger.warning(f"T+0验证失败: {e}")
        lines.append(f"_验证数据获取失败: {e}_\n")

    # ============================================================
    # 4. 选股命中统计（两周周期追踪 + 累计命中）
    # ============================================================
    lines.append("\n## 四、选股命中统计\n")
    try:
        summary = get_tracking_summary("pre_market")
        if len(summary) > 0:
            # 总览
            total_cycles = summary["total_cycles"].sum() if "total_cycles" in summary.columns else 0
            total_cum = summary["cumulative_hits"].sum() if "cumulative_hits" in summary.columns else 0
            active_cyc = summary[summary["active_cycle_end"] >= datetime.now().strftime("%Y-%m-%d")] if "active_cycle_end" in summary.columns else pd.DataFrame()
            lines.append(f"- 累计追踪: {len(summary)} 只股票, {total_cum} 次命中, {total_cycles} 个周期")
            lines.append(f"- 活跃周期: {len(active_cyc)} 只正在追踪中")
            hit_data = {
                "total_stocks": int(len(summary)),
                "total_hits": int(total_cum),
                "total_cycles": int(total_cycles),
                "active_count": int(len(active_cyc)),
                "multi_hit_count": 0,
                "multi_hit_codes": [],
                "pre_top5": [],
                "post_top5": [],
            }
            if len(active_cyc) > 0 and "active_cycle_hits" in active_cyc.columns:
                multi_hit = active_cyc[active_cyc["active_cycle_hits"] >= 2]
                if len(multi_hit) > 0:
                    lines.append(f"- 周期内多次命中: {len(multi_hit)} 只 (高频信号)")
                    lines.append(f"  高频股票: {', '.join(multi_hit['code'].head(5).tolist())}")
                    hit_data["multi_hit_count"] = int(len(multi_hit))
                    hit_data["multi_hit_codes"] = multi_hit["code"].head(5).tolist()

            # 盘前累计Top5
            pre = get_tracking_summary("pre_market")
            post = get_tracking_summary("post_market")
            if len(pre) > 0:
                lines.append("\n### 盘前累计命中 Top 5\n")
                lines.append("| 代码 | 累计命中 | 周期数 | 当前周期内 | 最近命中 |")
                lines.append("|------|----------|--------|-----------|----------|")
                for _, r in pre.head(5).iterrows():
                    lines.append(f"| {r['code']} | {r['cumulative_hits']} | {r['total_cycles']} | {r['active_cycle_hits']} | {r['last_pick_date']} |")
                    hit_data["pre_top5"].append({
                        "code": r["code"], "cumulative_hits": int(r["cumulative_hits"]),
                        "total_cycles": int(r["total_cycles"]),
                        "active_cycle_hits": int(r["active_cycle_hits"]),
                        "last_pick_date": r["last_pick_date"],
                    })
            if len(post) > 0:
                lines.append("\n### 盘后累计命中 Top 5\n")
                lines.append("| 代码 | 累计命中 | 最近命中 |")
                lines.append("|------|----------|----------|")
                for _, r in post.head(5).iterrows():
                    lines.append(f"| {r['code']} | {r['cumulative_hits']} | {r['last_pick_date']} |")
                    hit_data["post_top5"].append({
                        "code": r["code"], "cumulative_hits": int(r["cumulative_hits"]),
                        "last_pick_date": r["last_pick_date"],
                    })
            data["hit_stats"] = hit_data
        else:
            lines.append("_暂无命中追踪数据_\n")
    except Exception as e:
        logger.warning(f"命中统计失败: {e}")
        lines.append("_统计数据暂不可用_\n")

    lines.append(f"\n---\n")
    lines.append(f"*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # ============================================================
    # 5. 命中追踪
    # ============================================================
    try:
        lines.append("\n")
        tracking = get_tracking_report()
        lines.append(tracking)
        data["tracking_md"] = tracking
    except Exception as e:
        logger.warning(f"命中追踪报告失败: {e}")

    return "\n".join(lines), data


def _batch_preload_prices(codes: list[str], config: dict,
                          price_loader) -> dict[str, list[float]]:
    """
    批量预取近20日收盘价。优先 DB，缺失用顺序 API 补全并写回 DB。
    返回: {code: [close_prices_last_20_days]}
    """
    from database import get_market_db

    mdb = get_market_db()
    results = {}

    # Step 1: 批量 SQL 查询
    batch_size = 500
    db_hit = 0
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = mdb.conn.execute(f"""
            SELECT code, close FROM daily_price
            WHERE substr(code, 1, 6) IN ({placeholders})
            ORDER BY code, date DESC
        """, batch).fetchall()
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in rows:
            code = r["code"].split(".")[0].zfill(6) if "." in r["code"] else r["code"]
            grouped[code].append(r["close"])
        for code, closes in grouped.items():
            if len(closes) >= 20:
                results[code] = closes[:20]
                db_hit += 1

    missing = [c for c in codes if c not in results]
    if not missing:
        logger.info(f"K线预取: {db_hit}/{len(codes)} DB命中, 无需API补全")
        return results

    # Step 2: 顺序 API 补全（CPU-safe + SQLite 线程安全）
    api_hit = 0
    for code in missing:
        try:
            df = price_loader.get_price(code, days=20)
            if len(df) >= 20 and "close" in df.columns:
                closes = df["close"].tolist()
                # 写回 DB
                rows_to_insert = []
                _sfx = "SH" if code.startswith(("6", "9")) else ("SZ" if code.startswith(("0", "2", "3")) else "BJ")
                for _, r in df.iterrows():
                    date = str(r.get("date", ""))[:10]
                    rows_to_insert.append((
                        f"{code}.{_sfx}", date,
                        float(r.get("close", 0)),
                        float(r.get("change_pct", 0)) if pd.notna(r.get("change_pct")) else 0,
                        float(r.get("volume", 0)) if pd.notna(r.get("volume")) else 0,
                        float(r.get("amount", 0)) if pd.notna(r.get("amount")) else 0,
                    ))
                if rows_to_insert:
                    try:
                        mdb.bulk_insert_prices(rows_to_insert)
                    except Exception:
                        pass
                results[code] = closes[:20]
                api_hit += 1
        except Exception:
            pass

    logger.info(f"K线预取完成: DB={db_hit} API={api_hit} 总计={len(results)}/{len(codes)}")
    return results


def _analyze_cause(row, prices: dict, code: str) -> list[str]:
    """分析个股上涨动因（使用预取的价格数据，无网络调用）。"""
    causes = []
    # 量比
    vol_ratio = row.get("volume_ratio")
    if pd.notna(vol_ratio) and vol_ratio > 1.5:
        causes.append(f"放量(量比{vol_ratio:.1f})")
    # 振幅
    amp = row.get("amplitude")
    if pd.notna(amp) and amp > 5:
        causes.append(f"高波动(振幅{amp:.1f}%)")
    # 技术形态（从预取价格计算）
    close = prices.get(code)
    if close and len(close) >= 20:
        # 突破近期高点
        if close[0] >= max(close[:20]) * 0.98:
            causes.append("突破近期高点")
        # 连续阳线（价格从旧到新排列，取最后3根）
        if len(close) >= 4 and close[0] > close[1] > close[2]:
            causes.append("连阳走势")
    if not causes:
        sector = row.get("sector", "")
        causes.append(f"{sector}板块联动" if sector else "资金推动")
    return causes


def _compute_stop_loss(closes: list[float], close_today: float) -> float | None:
    """ATR 代理止损价：用近 N 日收盘价收益率估算日均波动，止损 = 收盘 × (1 - 2×日均波动)。

    采用收益率（比例）而非绝对价差，规避预取 K 线（可能后复权/滞后）与实时价混用的尺度错配。
    """
    if not closes or len(closes) < 5 or close_today is None or close_today <= 0:
        return None
    try:
        rets = []
        for i in range(len(closes) - 1):
            if closes[i + 1] and closes[i + 1] > 0:
                rets.append(abs(closes[i] / closes[i + 1] - 1))
        if not rets:
            return None
        vol = sum(rets) / len(rets)  # 日均绝对收益率
        if vol <= 0:
            return None
        stop = close_today * (1 - 2 * vol)
        return round(max(stop, 0.01), 2)
    except Exception:
        return None


# ============================================================
# HTML 渲染（复用 html_report.py 的 Qbot 风格范式）
# ============================================================

_HTML_CSS = """
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f7fa;color:#333;padding:20px;max-width:1200px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:24px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:22px;margin-bottom:8px}}
.header .meta{{opacity:.85;font-size:14px}}
.summary{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}}
.card{{background:#fff;border-radius:10px;padding:16px 20px;flex:1;min-width:160px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.card .label{{font-size:12px;color:#999;margin-bottom:4px}}
.card .value{{font-size:24px;font-weight:700}}
.card.ok{{border-left:3px solid #67c23a}}
.card.info{{border-left:3px solid #409eff}}
.card.warn{{border-left:3px solid #f56c6c}}
.charts{{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:20px}}
.chart-box{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.06);height:320px}}
.section{{background:#fff;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section h2{{font-size:18px;margin-bottom:12px;color:#667eea}}
.section h3{{font-size:15px;margin:16px 0 8px;color:#409eff}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f0f2f5;padding:8px 6px;text-align:left;font-weight:600;border-bottom:2px solid #e4e7ed;white-space:nowrap}}
td{{padding:6px;border-bottom:1px solid #ebeef5}}
tr:hover{{background:#f5f7fa}}
.up{{color:#f56c6c;font-weight:600}}
.down{{color:#67c23a;font-weight:600}}
.tag{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:600;background:#ecf5ff;color:#409eff}}
.footer{{text-align:center;color:#999;font-size:12px;padding:16px}}
.track-h2{{font-size:17px;color:#764ba2;margin:18px 0 10px;border-left:3px solid #764ba2;padding-left:10px}}
.track-h3{{font-size:15px;color:#409eff;margin:14px 0 8px}}
ul{{margin:6px 0 6px 20px}}
li{{margin:3px 0}}
"""

_CHART_JS = """
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
"""


def _esc(val) -> str:
    """HTML 转义。"""
    s = str(val)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fmt_chg(val) -> str:
    if val is None:
        return "-"
    return f"{val:+.1f}%" if val >= 0 else f"{val:.1f}%"


def _md_block_to_html(md: str) -> str:
    """
    轻量级 Markdown -> HTML 转换，仅支持本复盘命中追踪部分用到的语法:
      #~#### 标题、| 表格 |、无序列表、普通段落。
    """
    if not md:
        return ""
    lines = md.splitlines()
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            cls = f"track-h{lvl}" if lvl >= 2 else ""
            out.append(f"<h{lvl} class='{cls}'>{_esc(m.group(2))}</h{lvl}>")
            i += 1
            continue
        # 表格（连续 | 行）
        if line.startswith("|"):
            tbl = []
            while i < n and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            rows = []
            for r in tbl:
                if re.match(r"^\|[\s:|\-]+\|$", r):   # 分隔行
                    continue
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                rows.append(cells)
            if len(rows) >= 1:
                html = "<table><tr>" + "".join(f"<th>{_esc(c)}</th>" for c in rows[0]) + "</tr>"
                for r in rows[1:]:
                    tds = "".join(f"<td>{_esc(c)}</td>" for c in r)
                    html += f"<tr>{tds}</tr>"
                html += "</table>"
                out.append(html)
            continue
        # 无序列表
        if re.match(r"^[-*]\s+", line):
            out.append("<ul>")
            while i < n and re.match(r"^[-*]\s+", lines[i].lstrip()):
                item = re.sub(r"^[-*]\s+", "", lines[i].strip())
                out.append(f"<li>{_esc(item)}</li>")
                i += 1
            out.append("</ul>")
            continue
        # 段落
        out.append(f"<p>{_esc(line)}</p>")
        i += 1
    return "\n".join(out)


def _cards_html(data: dict) -> str:
    top = data.get("top_sectors", [])
    lead_sector = top[0]["sector"] if top else "-"
    t0 = data.get("t0_verify")
    by_cat = (t0 or {}).get("by_cat", {}) if t0 else {}
    t0_count = sum(len(v) for v in by_cat.values())
    hit = data.get("hit_stats") or {}
    cards = [
        ("最强板块", lead_sector, "info"),
        ("上榜板块", f"{len(top)} 个", "ok"),
        ("当日表现", f"{t0_count} 只", "warn" if t0_count else "info"),
        ("累计命中", f"{hit.get('total_hits', 0)} 次", "ok"),
    ]
    html = '<div class="summary">'
    for label, value, cls in cards:
        html += f'<div class="card {cls}"><div class="label">{label}</div><div class="value">{_esc(value)}</div></div>'
    html += "</div>"
    return html


def _sectors_table_html(data: dict) -> str:
    rows = data.get("top_sectors", [])
    if not rows:
        return '<p style="color:#999">暂无板块数据</p>'
    html = '<table><tr><th>板块</th><th>平均涨幅</th><th>股票数</th><th>龙头涨幅</th><th>成交额(亿)</th></tr>'
    for r in rows:
        avg = _fmt_chg(r.get("avg_chg"))
        avg_cls = "up" if (r.get("avg_chg") or 0) >= 0 else "down"
        mx = _fmt_chg(r.get("max_chg"))
        mx_cls = "up" if (r.get("max_chg") or 0) >= 0 else "down"
        html += (f'<tr><td><b>{_esc(r["sector"])}</b></td>'
                 f'<td class="{avg_cls}">{avg}</td>'
                 f'<td>{r.get("stock_count","-")}</td>'
                 f'<td class="{mx_cls}">{mx}</td>'
                 f'<td>{r.get("amount_yi","-")}</td></tr>')
    html += "</table>"
    return html


def _sector_stocks_html(data: dict) -> str:
    sec = data.get("sector_stocks", {})
    if not sec:
        return '<p style="color:#999">暂无个股数据</p>'
    html = ""
    for name, stocks in sec.items():
        html += f'<h3>{_esc(name)}</h3>'
        if not stocks:
            html += '<p style="color:#999">个股数据不足</p>'
            continue
        html += '<table><tr><th>代码</th><th>名称</th><th>股价</th><th>一手价</th><th>止损</th><th>涨幅</th><th>量比</th><th>成交额(亿)</th><th>动因</th></tr>'
        for s in stocks:
            chg = s.get("chg")
            chg_cls = "up" if (chg or 0) >= 0 else "down"
            html += (f'<tr><td>{_esc(s.get("code","-"))}</td>'
                     f'<td>{_esc(s.get("name","-"))}</td>'
                     f'<td>{s.get("close","-") if s.get("close") is not None else "-"}</td>'
                     f'<td>{s.get("lot","-") if s.get("lot") is not None else "-"}</td>'
                     f'<td>{s.get("stop_loss","-") if s.get("stop_loss") is not None else "-"}</td>'
                     f'<td class="{chg_cls}">{_fmt_chg(chg)}</td>'
                     f'<td>{s.get("vol_ratio","-") if s.get("vol_ratio") is not None else "-"}</td>'
                     f'<td>{s.get("amount_yi","-") if s.get("amount_yi") is not None else "-"}</td>'
                     f'<td>{" ".join(f"<span class=\'tag\'>{_esc(c)}</span>" for c in s.get("causes", []))}</td></tr>')
        html += "</table>"
    return html


def _t0_html(data: dict) -> str:
    t0 = data.get("t0_verify")
    if not t0:
        return '<p style="color:#999">暂无早盘选股记录</p>'
    html = f'<p style="color:#666">早盘 run_id={t0.get("run_id","-")}, {t0.get("date","-")}</p>'
    by_cat = t0.get("by_cat", {})
    if not by_cat:
        html += '<p style="color:#999">无验证明细</p>'
        return html
    for cat, rows in by_cat.items():
        html += f'<h3>{_esc(cat)}</h3>'
        if not rows:
            html += '<p style="color:#999">-</p>'
            continue
        html += '<table><tr><th>代码</th><th>名称</th><th>选股评分</th><th>当日表现</th><th>建议</th></tr>'
        for r in rows:
            t0p = r.get("t0", "-")
            t0cls = "up" if (isinstance(t0p, str) and t0p.startswith("+")) else ("down" if (isinstance(t0p, str) and t0p.startswith("-")) else "")
            advice = r.get("advice", "-")
            adv_cls = "up" if "符合" in advice else ("down" if "偏弱" in advice else "")
            html += (f'<tr><td>{_esc(r.get("code","-"))}</td>'
                     f'<td>{_esc(r.get("name","-"))}</td>'
                     f'<td>{r.get("score","-")}</td>'
                     f'<td class="{t0cls}">{_esc(t0p)}</td>'
                     f'<td class="{adv_cls}">{_esc(advice)}</td></tr>')
        html += "</table>"
    return html


def _hit_html(data: dict) -> str:
    hit = data.get("hit_stats")
    if not hit:
        return '<p style="color:#999">暂无命中追踪数据</p>'
    bullets = [
        f"累计追踪: {hit.get('total_stocks',0)} 只股票, {hit.get('total_hits',0)} 次命中, {hit.get('total_cycles',0)} 个周期",
        f"活跃周期: {hit.get('active_count',0)} 只正在追踪中",
    ]
    if hit.get("multi_hit_count"):
        bullets.append(f"周期内多次命中: {hit.get('multi_hit_count')} 只 (高频信号)")
        bullets.append(f"高频股票: {', '.join(map(str, hit.get('multi_hit_codes', [])))}")
    html = "<ul>" + "".join(f"<li>{_esc(b)}</li>" for b in bullets) + "</ul>"

    def top5_table(rows, cols, keys):
        head = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        body = ""
        for r in rows:
            body += "<tr>" + "".join(f"<td>{_esc(r.get(k,'-'))}</td>" for k in keys) + "</tr>"
        return f"<table>{head}{body}</table>"

    pre = hit.get("pre_top5", [])
    if pre:
        html += "<h3>盘前累计命中 Top 5</h3>"
        html += top5_table(pre,
                           ["代码", "累计命中", "周期数", "当前周期内", "最近命中"],
                           ["code", "cumulative_hits", "total_cycles", "active_cycle_hits", "last_pick_date"])
    post = hit.get("post_top5", [])
    if post:
        html += "<h3>盘后累计命中 Top 5</h3>"
        html += top5_table(post,
                           ["代码", "累计命中", "最近命中"],
                           ["code", "cumulative_hits", "last_pick_date"])
    return html


def generate_review_html(data: dict) -> str:
    """生成自包含 HTML 盘后复盘报告（复用 html_report 的 Qbot 风格）。"""
    if data is None:
        data = {}
    today = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    gen_at = data.get("generated_at", "")

    # ECharts 板块柱状图数据
    top = data.get("top_sectors", [])
    sector_names = [r["sector"] for r in top]
    sector_chgs = [r.get("avg_chg") or 0 for r in top]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>盘后复盘报告 — {today}</title>
{_CHART_JS}
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="header">
  <h1>📊 盘后复盘报告 — {today}</h1>
  <div class="meta">生成时间: {gen_at} | 数据来源: a-stock-engine afternoon_review</div>
</div>

{_cards_html(data)}

<div class="charts">
  <div class="chart-box"><div id="chartSector" style="width:100%;height:100%"></div></div>
</div>

<div class="section">
  <h2>🏆 一、今日最强板块 Top 5</h2>
  {_sectors_table_html(data)}
</div>

<div class="section">
  <h2>🔥 二、各板块最强个股</h2>
  {_sector_stocks_html(data)}
</div>

<div class="section">
  <h2>✅ 三、早盘推荐当日表现</h2>
  {_t0_html(data)}
</div>

<div class="section">
  <h2>📈 四、选股命中统计</h2>
  {_hit_html(data)}
</div>

<div class="section">
  <h2>🎯 五、选股命中追踪</h2>
  {_md_block_to_html(data.get("tracking_md", ""))}
</div>

<div class="footer">🚀 自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")} | Powered by a-stock-engine</div>

<script>
(function(){{
  var c = echarts.init(document.getElementById('chartSector'));
  c.setOption({{
    title: {{text:'板块平均涨幅 Top 5', left:'center', textStyle:{{fontSize:14}}}},
    tooltip: {{trigger:'axis', formatter:'{{b}}<br/>平均涨幅: {{c}}%'}},
    grid: {{left:60, right:30, top:40, bottom:60}},
    xAxis: {{type:'category', data:{sector_names!r}, axisLabel:{{interval:0, rotate:20, fontSize:11}}}},
    yAxis: {{type:'value', axisLabel:{{formatter:'{{value}}%'}}}},
    series: [{{type:'bar', data:{sector_chgs!r}, barWidth:'45%',
      itemStyle:{{color:new echarts.graphic.LinearGradient(0,0,0,1,[
        {{offset:0,color:'#f56c6c'}},{{offset:1,color:'#764ba2'}}])}},
      label:{{show:true, position:'top', formatter:'{{c}}%', fontSize:11}}}}]
  }});
  window.addEventListener('resize', function(){{ c.resize(); }});
}})();
</script>
</body></html>
"""
    return html


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

    # 加载配置
    import yaml
    config_path = Path(__file__).parent.parent / "config/config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    try:
        # 预取全市场数据（用于板块聚合 + T+0验证 + 盘后追踪）
        all_stocks = cli.get_stock_list()
        # 归一化 code 为 6 位字符串（缓存 round-trip 可能丢失前导零）
        if len(all_stocks) > 0 and "code" in all_stocks.columns:
            all_stocks["code"] = all_stocks["code"].astype(str).str.zfill(6)
        # 先注入 westock 实时行情，确保 change_pct 可用后再预取 K 线（否则 nlargest 拿到乱序代码）
        all_stocks = _inject_westock_quotes(all_stocks)
        codes_for_prices = []
        if len(all_stocks) > 0 and "code" in all_stocks.columns:
            # 预取范围扩大到 top 300，避免涨停股并列(+20%)时把板块领涨股挤出预取列表
            codes_for_prices = all_stocks.nlargest(300, "change_pct")["code"].tolist() if "change_pct" in all_stocks.columns else all_stocks["code"].head(300).tolist()
        preloaded_prices = _batch_preload_prices(codes_for_prices, config, price_loader)

        content, review_data = review_sectors(cli, price_loader, all_stocks, preloaded_prices, config)

        # 保存 Markdown（按日期分目录；带分钟时间戳归档 + 固定名指针双写）
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path("history") / today
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%H%M")
        # 1) 带分钟时间戳的归档文件（唯一，多次运行不覆盖）
        md_ts = save_dir / f"盘后复盘报告_{ts}.md"
        md_ts.write_text(content, encoding="utf-8")
        logger.info(f"复盘报告已归档(带时间戳): {md_ts}")
        # 2) 固定名指针文件（始终=最新一次，向后兼容）
        md_path = save_dir / "盘后复盘报告.md"
        md_path.write_text(content, encoding="utf-8")

        # 保存 HTML（新增，便于阅读；归档 + 指针双写）
        try:
            html = generate_review_html(review_data)
            html_ts = save_dir / f"盘后复盘报告_{ts}.html"
            html_ts.write_text(html, encoding="utf-8")
            logger.info(f"复盘HTML已归档(带时间戳): {html_ts}")
            html_path = save_dir / "盘后复盘报告.html"
            html_path.write_text(html, encoding="utf-8")
            logger.info(f"复盘HTML已保存: {html_path}")
        except Exception as e:
            logger.warning(f"HTML 生成失败（不影响 Markdown）: {e}")

        # 记录盘后命中追踪
        try:
            if len(all_stocks) > 0 and "change_pct" in all_stocks.columns:
                top_gainers = all_stocks.nlargest(30, "change_pct")
                if "code" in top_gainers.columns:
                    # 盘后追踪 = 当日涨幅前30强势股观察(非买入候选), 标注观察类目
                    # 避免未来再写 NULL(与历史补标 ④盘后强势股观察 一致)
                    track_picks(top_gainers, session_type="post_market",
                                category="④盘后强势股观察")
        except Exception as e:
            logger.debug(f"盘后追踪记录跳过: {e}")

        print(f"\n复盘报告: {md_path}")
    except Exception as e:
        logger.error(f"复盘失败: {e}", exc_info=True)
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
