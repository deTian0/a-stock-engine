"""
collect_pit_fundamentals.py — PIT 基本面采集器(时点正确, 消除前视)

与 collect_fundamentals.py(静态最新截面)不同, 本采集器落地"带公告日的全期财报"
和"逐交易日估值", 供回测做 Point-in-Time 时点查询:

  1) 财务: fina_indicator 全期(不传 period) + ann_date(公告日)。
     存 fundamentals_pit(code, end_date, ann_date, ...)。
     回测时对日 T 只取 ann_date <= T 的最近报告期 -> 杜绝"用未来财报"。
  2) 估值: daily_basic 逐交易日 trade_date 拉全市场, 存 daily_basic_pit。
     回测时取 trade_date = T 的 pe/pb/total_mv -> 杜绝"用最新估值"。

DB-first 增量: 已入库的 (code,end_date) / trade_date 自动跳过, 可反复重跑续传。
CPU 安全: 全程顺序, 调用间 sleep, 无并发。

用法
----
    python -m src.collect_pit_fundamentals                 # 仅财务(默认)
    python -m src.collect_pit_fundamentals --valuation     # 财务 + 估值(逐日, 重)
    python -m src.collect_pit_fundamentals --codes 000001,600519  # 子集调试
    python -m src.collect_pit_fundamentals --financials-only --dry-run

注意
----
  - 估值采集对 ~2700 个交易日各一次 daily_basic 调用, 属重量级, 默认关闭;
    未采集时回测自动用 close(T) ÷ PIT eps_ttm/bps 推导(同样 PIT 正确)。
  - 依赖 TUSHARE_TOKEN(15000 积分), fina_indicator/daily_basic 为付费接口。
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 确保本目录(src)在 path 中, 使 `python -m src.collect_pit_fundamentals` 能找到同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from tushare_provider import get_tushare, _ts_code, _from_ts_code
from database import get_market_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_pit_fundamentals")

# CPU 安全: 调用间最小间隔
CALL_SLEEP = 0.12          # 秒 (Tushare 500 req/min 下限附近)
FIN_BATCH = 500            # fina_indicator 每批股票数
VAL_LIMIT = 6000           # daily_basic 每交易日分页上限(全市场>5000只)


# ------------------------------------------------------------
#  财务: 全期 + 公告日
# ------------------------------------------------------------
def fetch_fina_all(pro, ts_codes: list[str], existing: set) -> pd.DataFrame:
    """fina_indicator 不加 period -> 返回该批股票所有报告期(含 ann_date)。

    **关键修复(2026-08-11)**: fina_indicator 对多只批量查询默认按 end_date 降序返回,
    且单次要约 3000 行上限截断 -> 一次性查 500 只只会拿回最近的 ~6 个季度, 早期历史
    (2017-2021) 被砍掉, 导致回测早期无 PIT 财务可用。
    改为 limit/offset 自动翻页(limit=5000, offset 步进), 翻到空页为止, 拉全历史。
    DB-first: 已存在的 (code,end_date) 跳过(2022-2026 已全, 只补 2017-2021 缺口)。
    """
    fields = ("ts_code,ann_date,end_date,roe,roa,grossprofit_margin,debt_to_assets,"
              "or_yoy,netprofit_yoy,eps,eps_ttm,bps,report_type")
    PAGE = 5000
    frames = []
    skipped = 0
    total_batches = (len(ts_codes) + FIN_BATCH - 1) // FIN_BATCH
    for b in range(total_batches):
        chunk = ",".join(ts_codes[b * FIN_BATCH:(b + 1) * FIN_BATCH])
        chunk_frames = []
        for off in range(0, 100000, PAGE):
            try:
                df = pro.fina_indicator(ts_code=chunk, fields=fields,
                                        limit=PAGE, offset=off)
            except Exception as e:
                logger.warning(f"fina_indicator 批次 {b} off {off} 失败: {e}")
                time.sleep(CALL_SLEEP)
                break
            if df is None or len(df) == 0:
                break
            chunk_frames.append(df)
            if len(df) < PAGE:
                break
            time.sleep(CALL_SLEEP)
        if not chunk_frames:
            time.sleep(CALL_SLEEP)
            continue
        df = pd.concat(chunk_frames, ignore_index=True)
        df["code"] = df["ts_code"].apply(_from_ts_code)
        df = df.rename(columns={
            "grossprofit_margin": "gross_margin",
            "debt_to_assets": "debt_ratio",
            "or_yoy": "revenue_growth",
            "netprofit_yoy": "profit_growth",
        })
        # DB-first 过滤
        before = len(df)
        mask = df.apply(lambda r: (r["code"], str(r["end_date"])) not in existing, axis=1)
        df = df[mask]
        skipped += before - len(df)
        if len(df) > 0:
            frames.append(df)
        if (b + 1) % 2 == 0:
            logger.info(f"  财务批次进度 {b+1}/{total_batches}, 累计新数据 {sum(len(f) for f in frames)}")
        time.sleep(CALL_SLEEP)
    if not frames:
        logger.info(f"fina_indicator: 无新数据(skipped {skipped})")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    keep = ["code", "end_date", "ann_date", "roe", "roa", "gross_margin",
            "debt_ratio", "revenue_growth", "profit_growth", "eps",
            "eps_ttm", "bps", "report_type"]
    combined = combined[[c for c in keep if c in combined.columns]]
    logger.info(f"fina_indicator 新数据: {len(combined)} 条 (skipped {skipped})")
    return combined


# ------------------------------------------------------------
#  估值: 逐交易日 daily_basic
# ------------------------------------------------------------
def collect_valuation(pro, db, trade_dates: list[str]) -> int:
    """逐交易日拉 daily_basic, 入库 daily_basic_pit。DB-first 续传。"""
    existing = db.existing_pit_val_dates()
    pending = [d for d in trade_dates if d not in existing]
    logger.info(f"估值采集: 总 {len(trade_dates)} 日, 待采 {len(pending)} 日")
    total = 0
    for idx, d in enumerate(pending):
        try:
            frames = []
            for off in range(0, VAL_LIMIT, 3000):
                df = pro.daily_basic(
                    trade_date=d.replace("-", ""),
                    fields="ts_code,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,total_mv,circ_mv",
                    offset=off, limit=3000,
                )
                if df is None or len(df) == 0:
                    break
                frames.append(df)
                if len(df) < 3000:
                    break
            if not frames:
                time.sleep(CALL_SLEEP)
                continue
            val = pd.concat(frames, ignore_index=True)
            val["code"] = val["ts_code"].apply(_from_ts_code)
            for col in ("total_mv", "circ_mv"):
                if col in val.columns:
                    val[col] = val[col] * 1e4          # 万元 -> 元
            val["trade_date"] = d
            n = db.upsert_daily_basic_pit(val)
            total += n
        except Exception as e:
            logger.warning(f"daily_basic {d} 失败: {e}")
        if (idx + 1) % 100 == 0:
            logger.info(f"  估值进度: {idx+1}/{len(pending)}, 已入库 {total}")
        time.sleep(CALL_SLEEP)
    logger.info(f"估值采集完成: 本次新增 {total} 条")
    return total


# ------------------------------------------------------------
#  主流程
# ------------------------------------------------------------
def collect(pro, db, do_fin: bool = True, do_val: bool = False,
            codes_filter: list[str] = None) -> dict:
    result = {"fin": 0, "val": 0}

    if do_fin:
        # 基础列表 -> ts_code
        basic = pro.stock_basic(exchange="", list_status="L",
                                fields="ts_code,symbol,name,area,industry,market,list_date")
        basic["code"] = basic["ts_code"].apply(_from_ts_code)
        if codes_filter:
            cf = [c.zfill(6) for c in codes_filter]
            basic = basic[basic["code"].isin(cf)].copy()
        ts_codes = basic["ts_code"].tolist()
        logger.info(f"财务采集: {len(ts_codes)} 只")
        existing = db.existing_pit_fin_keys()
        fin = fetch_fina_all(pro, ts_codes, existing)
        if len(fin) > 0:
            result["fin"] = db.upsert_fundamentals_pit(fin)
            logger.info(f"已 upsert 财务 {result['fin']} 条 -> fundamentals_pit")

    if do_val:
        # 交易日历取自 daily_price(回测实际用到的日期)
        rows = db.conn.execute(
            "SELECT DISTINCT date FROM daily_price ORDER BY date"
        ).fetchall()
        trade_dates = [r[0] for r in rows if r[0]]
        logger.info(f"估值采集: {len(trade_dates)} 个交易日")
        result["val"] = collect_valuation(pro, db, trade_dates)

    return result


def main():
    parser = argparse.ArgumentParser(description="PIT 基本面采集器(时点正确)")
    parser.add_argument("--financials", dest="fin", action="store_true",
                        default=True, help="采集财报(默认开)")
    parser.add_argument("--no-financials", dest="fin", action="store_false",
                        help="跳过财报")
    parser.add_argument("--valuation", action="store_true",
                        help="采集逐交易日估值(重量级, 默认关)")
    parser.add_argument("--codes", default="", help="子集代码, 逗号分隔")
    parser.add_argument("--dry-run", action="store_true", help="只拉不写库")
    args = parser.parse_args()

    codes_filter = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    pro = get_tushare().pro
    db = get_market_db()

    if args.dry_run:
        logger.info("dry-run: 不写库")
        # dry-run 仅验证财务拉取
        basic = pro.stock_basic(exchange="", list_status="L",
                                fields="ts_code,symbol,name")
        basic["code"] = basic["ts_code"].apply(_from_ts_code)
        if codes_filter:
            basic = basic[basic["code"].isin([c.zfill(6) for c in codes_filter])]
        fin = fetch_fina_all(pro, basic["ts_code"].tolist(), set())
        logger.info(f"dry-run 财务: {len(fin)} 条")
        return

    res = collect(pro, db, do_fin=args.fin, do_val=args.valuation,
                  codes_filter=codes_filter)
    logger.info(f"采集完成: 财务 {res['fin']} 条, 估值 {res['val']} 条")


if __name__ == "__main__":
    main()
