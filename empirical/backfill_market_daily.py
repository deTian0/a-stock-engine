"""回补 market.db / daily_price 的全市场日线缺口。

背景
----
daily_price 共 828 万行，但**全市场覆盖止于 2026-04-17**：
  2026-04-20 之后每日仅 15~420 行（按需单只拉取的零星股票 + 14 只 ETF）。
后果：盘后复盘「最近一周持仓追踪」按 `date >= pick_date` 找选股日基准价，
近期 picks 全部找不到基准 → 301 只推荐仅 8 只可验证、胜率显示 0.0%。

单位约定（已实测校准，见 --validate）
------------------------------------
daily_price 同一列混用两套约定，本脚本只写「带后缀股票行」，沿用其约定：
  code    : ts_code 原样（000001.SZ），与存量 720 万行一致
  pct_chg : 小数制（-0.0072 = -0.72%），= tushare pct_chg / 100
  vol     : 股（tushare 手 × 100）
  amount  : 元（tushare 千元 × 1000）

用法
----
  python empirical/backfill_market_daily.py --validate      # 用重叠日期实测校准单位
  python empirical/backfill_market_daily.py --dry-run       # 只列出待回补日期
  python empirical/backfill_market_daily.py --apply         # 执行回补

CPU-safe: 严格顺序执行，每次 API 调用间 sleep，无并发。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tushare_provider import get_tushare  # noqa: E402

MARKET_DB = str(ROOT / "data_cache" / "market.db")
FULL_MARKET_MIN_ROWS = 3000   # 低于此值视为当日覆盖不全
API_SLEEP = 0.5               # 每次调用间隔，CPU/接口双友好


def existing_counts(conn) -> dict[str, int]:
    return {d: n for d, n in conn.execute(
        "SELECT date, COUNT(*) FROM daily_price GROUP BY date")}


def trading_dates(pro, start: str, end: str) -> list[str]:
    """SSE 交易日历，返回 YYYY-MM-DD 列表。"""
    cal = pro.trade_cal(exchange="SSE",
                        start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""),
                        is_open="1")
    if cal is None or len(cal) == 0:
        return []
    ds = sorted(str(x) for x in cal["cal_date"].tolist())
    return [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in ds]


def fetch_day(pro, date: str) -> pd.DataFrame | None:
    """拉取单个交易日全市场日线，返回归一化后的 DataFrame。"""
    raw = pro.daily(trade_date=date.replace("-", ""))
    if raw is None or len(raw) == 0:
        return None
    raw = raw[~raw["ts_code"].str.endswith(".BJ")].copy()   # 剔除北交所
    if len(raw) == 0:
        return None
    out = pd.DataFrame({
        "code": raw["ts_code"],
        "date": date,
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        # tushare pct_chg 为百分数 → 存量股票行为小数制
        "pct_chg": pd.to_numeric(raw["pct_chg"], errors="coerce") / 100.0,
        # tushare vol 单位「手」→ 股
        "vol": pd.to_numeric(raw["vol"], errors="coerce") * 100.0,
        # tushare amount 单位「千元」→ 元
        "amount": pd.to_numeric(raw["amount"], errors="coerce") * 1000.0,
    })
    return out.dropna(subset=["close"])


def validate(conn, pro) -> int:
    """用已有全市场覆盖日实测校准单位换算，避免回补数据与存量口径不一致。"""
    ref = conn.execute(
        "SELECT date FROM daily_price GROUP BY date "
        "HAVING COUNT(*)>? ORDER BY date DESC LIMIT 1", (FULL_MARKET_MIN_ROWS,)
    ).fetchone()
    if not ref:
        print("找不到全市场覆盖日，无法校准。")
        return 1
    date = ref[0]
    print(f"校准基准日: {date}（存量全市场覆盖）")

    api = fetch_day(pro, date)
    if api is None:
        print("tushare 该日无数据，换个日期再试。")
        return 1
    api = api.set_index("code")

    rows = conn.execute(
        "SELECT code, close, pct_chg, vol, amount FROM daily_price "
        "WHERE date=? AND code LIKE '%.%' LIMIT 400", (date,)
    ).fetchall()

    stats = {k: [] for k in ("close", "pct_chg", "vol", "amount")}
    n = 0
    for code, close, pct, vol, amt in rows:
        if code not in api.index:
            continue
        a = api.loc[code]
        n += 1
        for field, dbv in (("close", close), ("pct_chg", pct),
                           ("vol", vol), ("amount", amt)):
            av = float(a[field])
            if dbv is None:
                continue
            if abs(av) < 1e-12:
                stats[field].append(1.0 if abs(dbv) < 1e-12 else 0.0)
            else:
                stats[field].append(float(dbv) / av)

    if n == 0:
        print("无可比对的重叠代码。")
        return 1
    print(f"比对样本: {n} 只\n")
    print("  字段        存量/换算后API 比值中位数   判定")
    ok = True
    for field, vals in stats.items():
        if not vals:
            continue
        med = pd.Series(vals).median()
        good = abs(med - 1.0) < 0.02
        ok &= good
        print(f"  {field:10s} {med:>18.4f}   {'一致' if good else '不一致 <<<'}")
    print(f"\n结论: 单位换算{'正确，可回补' if ok else '有偏差，需修正后再回补'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-20")
    ap.add_argument("--end", default=None, help="默认今天")
    ap.add_argument("--validate", action="store_true", help="实测校准单位换算")
    ap.add_argument("--dry-run", action="store_true", help="只列出待回补日期")
    ap.add_argument("--apply", action="store_true", help="执行回补")
    args = ap.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(MARKET_DB)
    tp = get_tushare()
    pro = tp.pro

    if args.validate:
        rc = validate(conn, pro)
        conn.close()
        return rc

    counts = existing_counts(conn)
    dates = trading_dates(pro, args.start, end)
    if not dates:
        print("交易日历为空。")
        return 1
    todo = [d for d in dates if counts.get(d, 0) < FULL_MARKET_MIN_ROWS]

    print(f"区间 {args.start} ~ {end}: 交易日 {len(dates)} 个, "
          f"待回补 {len(todo)} 个")
    for d in todo:
        print(f"  {d}  现有 {counts.get(d, 0):5d} 行")

    if not args.apply:
        print("\n[DRY-RUN] 未写库。加 --apply 执行回补。")
        return 0

    total = 0
    failed = []
    for i, d in enumerate(todo, 1):
        try:
            df = fetch_day(pro, d)
            if df is None or len(df) == 0:
                failed.append((d, "空数据"))
                print(f"  [{i}/{len(todo)}] {d}  空数据，跳过")
                continue
            conn.executemany(
                "INSERT OR REPLACE INTO daily_price "
                "(code, date, close, pct_chg, vol, amount) VALUES (?,?,?,?,?,?)",
                df[["code", "date", "close", "pct_chg", "vol", "amount"]]
                .itertuples(index=False, name=None),
            )
            conn.commit()
            total += len(df)
            print(f"  [{i}/{len(todo)}] {d}  写入 {len(df)} 行")
        except Exception as e:
            failed.append((d, str(e)[:80]))
            print(f"  [{i}/{len(todo)}] {d}  失败: {e}")
        time.sleep(API_SLEEP)   # CPU-safe: 顺序 + 间隔

    print(f"\n回补完成: {total} 行, 失败 {len(failed)} 个日期")
    for d, why in failed:
        print(f"  {d}: {why}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
