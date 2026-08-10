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
import os
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


def review_sectors(cli, price_loader, all_stocks: pd.DataFrame = None,
                    preloaded_prices: dict = None, config: dict = None) -> str:
    """
    板块复盘: 获取今日最强板块，分析上涨动因。
    """
    if config is None:
        config = {}
    if all_stocks is None:
        all_stocks = pd.DataFrame()
    if preloaded_prices is None:
        preloaded_prices = {}
    lines = []
    lines.append(f"# 盘后复盘报告 — {datetime.now().strftime('%Y-%m-%d')}\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%H:%M')}\n")

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
        # 注入 westock-data 实时涨跌幅（tushare 不含 change_pct）
        if "change_pct" not in stock_list.columns or stock_list["change_pct"].isna().all():
            from westock_helpers import batch_change_pct
            chg_map = batch_change_pct(stock_list["code"].astype(str).str.zfill(6).tolist())
            if chg_map:
                stock_list["change_pct"] = stock_list["code"].astype(str).str.zfill(6).map(chg_map)
                logger.info(f"已注入westock涨跌幅: {stock_list['change_pct'].notna().sum()} 只")
        stock_list["sector"] = stock_list["code"].map(sector_mapping).fillna("综合")
        sector_agg = stock_list.groupby("sector").agg(
            avg_chg=("change_pct", "mean"),
            total_amt=("amount", "sum"),
            stock_count=("code", "count"),
            max_chg=("change_pct", "max"),
        ).reset_index()
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
            lines.append("| 代码 | 名称 | 当日股价 | 一手价格 | 涨幅 | 量比 | 成交额(亿) | 动因 |")
            lines.append("|------|------|---------|---------|------|------|-----------|------|")
            for _, row in top.iterrows():
                code = str(row.get("code", ""))
                name = str(row.get("name", code))
                if name.lower() in ("nan", "none", ""):
                    name = code
                # 当日股价 (从 close 列)
                close = row.get("close", 0)
                close_str = f"{close:.2f}" if pd.notna(close) and close > 0 else "-"
                lot_str = f"{close * 100:.0f}" if pd.notna(close) and close > 0 else "-"
                chg = f"{row.get('change_pct', 0):.1f}%" if pd.notna(row.get("change_pct")) else "-"
                vol_r = f"{row.get('volume_ratio', 0):.2f}" if pd.notna(row.get("volume_ratio")) else "-"
                amt = f"{row.get('amount', 0)/1e8:.1f}" if pd.notna(row.get("amount")) else "-"
                causes = _analyze_cause(row, preloaded_prices, code)
                lines.append(f"| {code} | {name} | {close_str} | {lot_str} | {chg} | {vol_r} | {amt} | {', '.join(causes[:2])} |")
        else:
            lines.append("_个股数据不足_\n")

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
            if len(active_cyc) > 0 and "active_cycle_hits" in active_cyc.columns:
                multi_hit = active_cyc[active_cyc["active_cycle_hits"] >= 2]
                if len(multi_hit) > 0:
                    lines.append(f"- 周期内多次命中: {len(multi_hit)} 只 (高频信号)")
                    lines.append(f"  高频股票: {', '.join(multi_hit['code'].head(5).tolist())}")

            # 盘前累计Top5
            pre = get_tracking_summary("pre_market")
            post = get_tracking_summary("post_market")
            if len(pre) > 0:
                lines.append("\n### 盘前累计命中 Top 5\n")
                lines.append("| 代码 | 累计命中 | 周期数 | 当前周期内 | 最近命中 |")
                lines.append("|------|----------|--------|-----------|----------|")
                for _, r in pre.head(5).iterrows():
                    lines.append(f"| {r['code']} | {r['cumulative_hits']} | {r['total_cycles']} | {r['active_cycle_hits']} | {r['last_pick_date']} |")
            if len(post) > 0:
                lines.append("\n### 盘后累计命中 Top 5\n")
                lines.append("| 代码 | 累计命中 | 最近命中 |")
                lines.append("|------|----------|----------|")
                for _, r in post.head(5).iterrows():
                    lines.append(f"| {r['code']} | {r['cumulative_hits']} | {r['last_pick_date']} |")
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
        lines.append(get_tracking_report())
    except Exception as e:
        logger.warning(f"命中追踪报告失败: {e}")

    return "\n".join(lines)


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
            WHERE code IN ({placeholders})
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
                for _, r in df.iterrows():
                    date = str(r.get("date", ""))[:10]
                    rows_to_insert.append((
                        f"{code}.SZ", date,
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
        codes_for_prices = []
        if len(all_stocks) > 0 and "code" in all_stocks.columns:
            codes_for_prices = all_stocks.nlargest(50, "change_pct")["code"].tolist() if "change_pct" in all_stocks.columns else all_stocks["code"].head(300).tolist()
        preloaded_prices = _batch_preload_prices(codes_for_prices, config, price_loader)

        content = review_sectors(cli, price_loader, all_stocks, preloaded_prices, config)

        # 保存
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path("history") / today
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "盘后复盘报告.md"
        save_path.write_text(content, encoding="utf-8")

        # 记录盘后命中追踪
        try:
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
