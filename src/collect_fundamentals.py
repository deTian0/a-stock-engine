"""
collect_fundamentals.py — 全市场基本面「尽量捞」采集器

从 Tushare 拉取全 A 股截面基本面并落地到 market.db 的 fundamentals 表：
  - 基础信息:  stock_basic        (name/industry/area/list_date/market)
  - 估值截面:  daily_basic        (pe/pe_ttm/pb/ps/ps_ttm/dv_ratio/total_mv/circ_mv)，最新交易日分页拉全市场
  - 财务截面:  fina_indicator     (roe/roa/毛利率/负债率/营收同比/净利同比/eps/bps)
               period=主期间(默认20260331, 覆盖率最高) + period=回填期间(默认20260630, 提早披露的公司覆盖)

数据质量修正（相对旧 get_fundamentals 的两个 bug）:
  - profit_growth 用 netprofit_yoy(净利润同比%) 而非 profit_dedt(扣非净利润绝对值,元)
  - market_cap(total_mv) 由 daily_basic 取，单位 万元 → ×1e4 转元

使用:
    python -m src.collect_fundamentals            # 全市场采集
    python -m src.collect_fundamentals --codes 000001,600519  # 子集(调试)
    python -m src.collect_fundamentals --dry-run  # 只拉不写库
    python -m src.collect_fundamentals --main-period 20260331 --backfill-period 20260630

注意: 依赖 TUSHARE_TOKEN (15000 积分)，fina_indicator 为付费接口；本脚本一次性拉全市场
约 5500 只 × 少量字段，积分消耗可控。后续刷新只需重跑（upsert 幂等）。
"""

import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from tushare_provider import get_tushare, _ts_code, _from_ts_code
from fundamental_store import FundamentalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_fundamentals")


# ------------------------------------------------------------
#  各数据源拉取
# ------------------------------------------------------------

def fetch_stock_basic(pro) -> pd.DataFrame:
    """基础信息（代码/名称/行业/地区/上市日/板块）。"""
    basic = pro.stock_basic(
        exchange="", list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,exchange"
    )
    if basic is None or len(basic) == 0:
        raise RuntimeError("stock_basic 返回空")
    basic["code"] = basic["ts_code"].apply(_from_ts_code)
    logger.info(f"stock_basic: {len(basic)} 只")
    return basic


def _latest_trade_date(pro) -> str:
    """探测最近一个有数据的交易日。"""
    for off in range(12):
        d = (datetime.now() - timedelta(days=off)).strftime("%Y%m%d")
        try:
            r = pro.daily_basic(trade_date=d, limit=1)
            if r is not None and len(r) > 0:
                return d
        except Exception:
            continue
    raise RuntimeError("无法探测最新交易日")


def fetch_daily_basic(pro) -> pd.DataFrame:
    """估值截面：分页拉全市场最新交易日。total_mv/circ_mv 单位 万元→元。"""
    trade_date = _latest_trade_date(pro)
    frames = []
    for off in range(0, 20000, 3000):
        df = pro.daily_basic(
            trade_date=trade_date,
            fields="ts_code,close,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,total_mv,circ_mv",
            offset=off, limit=3000,
        )
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if len(df) < 3000:
            break
    if not frames:
        raise RuntimeError("daily_basic 返回空")
    val = pd.concat(frames, ignore_index=True)
    val["code"] = val["ts_code"].apply(_from_ts_code)
    # 万元 → 元
    for col in ("total_mv", "circ_mv"):
        if col in val.columns:
            val[col] = val[col] * 1e4
    val["valuation_date"] = trade_date
    logger.info(f"daily_basic: {len(val)} 只 (交易日 {trade_date})")
    return val


def fetch_fina(pro, period: str, ts_codes: list[str]) -> pd.DataFrame:
    """财务截面：fina_indicator 按 period 过滤，批量拉取。返回 code 主键(每码最多1行)。"""
    if not ts_codes:
        return pd.DataFrame()
    fields = ("ts_code,roe,roa,grossprofit_margin,debt_to_assets,or_yoy,"
              "profit_dedt,netprofit_yoy,eps,bps,end_date")
    frames = []
    batch = 1000
    for i in range(0, len(ts_codes), batch):
        chunk = ",".join(ts_codes[i:i + batch])
        try:
            df = pro.fina_indicator(ts_code=chunk, period=period, fields=fields)
        except Exception as e:
            logger.warning(f"fina_indicator period={period} 批次 {i} 失败: {e}")
            continue
        if df is not None and len(df) > 0:
            frames.append(df)
    if not frames:
        logger.warning(f"fina_indicator period={period} 无数据返回")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["code"] = combined["ts_code"].apply(_from_ts_code)
    combined = combined.rename(columns={
        "grossprofit_margin": "gross_margin",
        "debt_to_assets": "debt_ratio",
        "or_yoy": "revenue_growth",
        "netprofit_yoy": "profit_growth",
        "profit_dedt": "dedt_net_profit",
        "end_date": "report_period",
    })
    keep = ["code", "roe", "roa", "gross_margin", "debt_ratio",
            "revenue_growth", "profit_growth", "dedt_net_profit",
            "eps", "bps", "report_period"]
    combined = combined[[c for c in keep if c in combined.columns]]
    # period 过滤下每码最多 1 行；保险去重保留第一条
    combined = combined.drop_duplicates(subset="code", keep="first")
    logger.info(f"fina_indicator period={period}: {len(combined)} 只")
    return combined


# ------------------------------------------------------------
#  合并 + 落库
# ------------------------------------------------------------

def collect(pro, main_period: str, backfill_period: str,
            codes_filter: list[str] = None, dry_run: bool = False) -> pd.DataFrame:
    """执行采集流程，返回合并后的全量 DataFrame（未写库时也可检查）。"""
    basic = fetch_stock_basic(pro)
    if codes_filter:
        cf = [c.zfill(6) for c in codes_filter]
        basic = basic[basic["code"].isin(cf)].copy()
        logger.info(f"子集过滤: {len(basic)} 只")

    # 估值
    val = fetch_daily_basic(pro)
    if codes_filter:
        val = val[val["code"].isin([c.zfill(6) for c in codes_filter])].copy()

    # 财务: 主期间
    ts_codes = basic["ts_code"].tolist()
    fin = fetch_fina(pro, main_period, ts_codes)

    # 财务: 回填期间（仅覆盖已披露公司，覆盖主期间）
    fin2 = fetch_fina(pro, backfill_period, ts_codes)

    # 合并基础信息 + 估值
    df = basic[[c for c in ["code", "name", "industry", "area",
                            "list_date", "market"] if c in basic.columns]].copy()
    df = df.merge(val, on="code", how="left")

    # 合并财务（主）
    if len(fin) > 0:
        df = df.merge(fin, on="code", how="left")

    # 回填覆盖（仅 fin2 中存在的 code 更新财务列，其余保留主期间值）
    if len(fin2) > 0:
        fin2_idx = fin2.set_index("code")
        fin_cols = [c for c in ["roe", "roa", "gross_margin", "debt_ratio",
                                "revenue_growth", "profit_growth", "dedt_net_profit",
                                "eps", "bps", "report_period"] if c in fin2.columns]
        for col in fin_cols:
            mapped = df["code"].map(fin2_idx[col])
            df[col] = mapped.where(mapped.notna(), df[col]) if col in df.columns else mapped

    logger.info(f"合并完成: {len(df)} 只, 列={list(df.columns)}")

    if dry_run:
        logger.info("dry-run 模式: 不写库")
        return df

    store = FundamentalStore()
    n = store.upsert(df)
    logger.info(f"已 upsert {n} 只到 fundamentals 表")
    return df


def main():
    parser = argparse.ArgumentParser(description="全市场基本面采集器")
    parser.add_argument("--main-period", default="20260331",
                        help="主财务期间(覆盖率最高), 默认 20260331(Q1 2026)")
    parser.add_argument("--backfill-period", default="20260630",
                        help="回填财务期间(提早披露公司), 默认 20260630(H1 2026)")
    parser.add_argument("--codes", default="", help="逗号分隔的子集代码, 如 000001,600519")
    parser.add_argument("--dry-run", action="store_true", help="只拉不写库")
    args = parser.parse_args()

    codes_filter = [c.strip() for c in args.codes.split(",") if c.strip()] or None

    pro = get_tushare().pro
    df = collect(pro, args.main_period, args.backfill_period,
                 codes_filter=codes_filter, dry_run=args.dry_run)

    # 覆盖率报告
    if not args.dry_run:
        store = FundamentalStore()
        store.coverage_report()
    else:
        # dry-run 也打印非空率
        total = len(df)
        logger.info(f"=== dry-run 覆盖率 (共 {total} 只) ===")
        for col in FundamentalStore().db.FUND_COLUMNS:
            if col in df.columns:
                nn = df[col].notna().sum()
                logger.info(f"  {col:16s}: {nn:5d} ({nn/total*100:5.1f}%)")


if __name__ == "__main__":
    main()
