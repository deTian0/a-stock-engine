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
from pick_tracker import get_tracking_report, track_picks

logger = logging.getLogger(__name__)


def _fetch_change_pct_westock(codes: list[str]) -> dict[str, float]:
    """用 westock-data batch kline 获取涨跌幅。分批调用，每批100只。返回 {code: change_pct%}。"""
    import subprocess
    if not codes:
        return {}
    
    changes = {}
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        ws_codes = ",".join(f"sh{s}" if s.startswith(("6","9")) else 
                            f"sz{s}" if s.startswith(("0","2","3")) else 
                            f"bj{s}" for s in batch)
        try:
            result = subprocess.run(
                f'npx -y westock-data-skillhub@1.0.5 kline {ws_codes} --period day --limit 2',
                capture_output=True, text=True, timeout=60, check=False, shell=True
            )
            if result.returncode != 0 or not result.stdout:
                continue
            latest, prev = {}, {}
            for line in result.stdout.strip().split("\n"):
                if not line or "|" not in line or "symbol" in line or "Batch" in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 8:
                    continue
                symbol = parts[1].replace("sh","").replace("sz","").replace("bj","").zfill(6)
                try:
                    close = float(parts[4])
                except ValueError:
                    continue
                if symbol not in latest:
                    latest[symbol] = close
                elif symbol not in prev:
                    prev[symbol] = close
            for code, today in latest.items():
                yesterday = prev.get(code, today)
                if yesterday and yesterday > 0:
                    changes[code] = round((today / yesterday - 1) * 100, 2)
        except Exception:
            continue

    logger.info(f"westock涨跌幅: {len(changes)}/{len(codes)} 只 ({len(changes) and 100*len(changes)//len(changes) or 0}%覆盖)")
    return changes


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
        # 注入 westock-data 实时涨跌幅（tushare 不含 change_pct）
        if "change_pct" not in stock_list.columns or stock_list["change_pct"].isna().all():
            sample_codes = stock_list["code"].astype(str).str.zfill(6).tolist()
            chg_map = _fetch_change_pct_westock(sample_codes)
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

        # T+2 验证: 找到≥2个自然日前选股且未验证的，用kline取T+0/T+2收盘价算收益
        try:
            db = get_db()
            today = datetime.now().strftime("%Y-%m-%d")
            # 找所有≥2天前的选股、且尚未在t2_verifications中
            rows = db.conn.execute("""
                SELECT DISTINCT s.code, DATE(s.pick_date) as pd
                FROM stock_picks s
                WHERE DATE(s.pick_date) <= DATE('now', '-2 days')
                AND NOT EXISTS (
                    SELECT 1 FROM t2_verifications v
                    WHERE v.code = s.code AND v.pick_date = DATE(s.pick_date)
                )
                LIMIT 50
            """).fetchall()

            if rows:
                from collections import defaultdict
                by_pick_date = defaultdict(list)
                for r in rows:
                    by_pick_date[r["pd"]].append(r["code"].zfill(6))

                import subprocess
                for pick_date, codes in by_pick_date.items():
                    # 计算从pick_date到今天的天数
                    delta = (datetime.now() - datetime.strptime(pick_date, "%Y-%m-%d")).days + 2
                    ws_codes = ",".join(f"sh{c}" if c.startswith(("6","9")) else f"sz{c}" for c in codes)
                    r = subprocess.run(
                        f"npx -y westock-data-skillhub@1.0.5 kline {ws_codes} --period day --limit {delta}",
                        capture_output=True, text=True, timeout=60, shell=True
                    )
                    if r.returncode != 0 or not r.stdout:
                        continue

                    # 解析: kline[0]=最新(today), kline[-1]=最旧(T+0)
                    by_code = defaultdict(list)
                    for line in r.stdout.strip().split("\n"):
                        if "|" not in line or "symbol" in line or "Batch" in line:
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) < 6:
                            continue
                        sym = parts[1].replace("sh","").replace("sz","").replace("bj","").zfill(6)
                        try:
                            by_code[sym].append(float(parts[4]))
                        except ValueError:
                            pass

                    verifications = []
                    for code in codes:
                        chain = by_code.get(code, [])
                        if len(chain) >= 2:
                            t2_close = chain[0]    # 今天收盘
                            t0_close = chain[-1]   # 最早(≈pick_date当天收盘)
                            ret = round((t2_close / t0_close - 1) * 100, 2) if t0_close > 0 else 0
                            status = "positive" if ret > 0 else "negative" if ret < 0 else "flat"
                            verifications.append({
                                "code": code, "name": code,
                                "t0_close": t0_close, "t2_close": t2_close,
                                "return_pct": ret, "status": status,
                            })
                    if verifications:
                        db.save_t2_verification(pick_date=pick_date, verifications=verifications)
                        logger.info(f"T+2验证入库: {pick_date} {len(verifications)}条")
        except Exception as e:
            logger.debug(f"T+2验证跳过: {e}")

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
