"""修复 post_market 命中追踪的「nlargest 退化」污染。

背景
----
afternoon_review 用 `all_stocks.nlargest(30, "change_pct")` 取当日涨幅前 30 强。
pandas 在排序列全为 NaN / 全为同一值时不报错，而是退化为「返回原始行序前 n 行」。
all_stocks 原始行序来自 tushare stock_basic（代码升序），因此当日 change_pct
未取到时，盘后强势股榜被写成 000001/000002/000006… 这类低位代码。
由于命中统计是累计的，污染永久留存在 picks.db。

判定
----
按 (pick_date, created_at 到分钟) 分批，若批内 >90% 的代码 < '002000'
且无任何 60/68/30 开头代码，则判定为退化批次。

动作
----
1. 删除退化批次的 pick_tracking 明细行；
2. 清空 post_market 的 pick_summary，按 pick_tracker 原始周期算法
   （14 自然日周期、同日不重复、周期内命中则延期并累加）重放存活明细，
   同时回填明细自身的 cycle_* / cumulative 字段。

pre_market 数据不受任何影响。默认 dry-run，加 --apply 才写库。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

DB = str(Path(__file__).resolve().parent.parent / "history" / "picks.db")
CYCLE_CALENDAR_DAYS = 14  # 与 src/pick_tracker.py 保持一致
SESSION = "post_market"


def _is_degenerate(codes: list[str]) -> bool:
    """批内代码是否呈「代码升序聚集」特征。"""
    if not codes:
        return False
    low = sum(1 for c in codes if c < "002000")
    cross = any(c.startswith(("60", "68", "30")) for c in codes)
    return (low / len(codes)) > 0.9 and not cross


def find_degenerate_batches(conn) -> list[tuple[str, str, list[int], list[str]]]:
    """返回 [(pick_date, batch_minute, [row_id...], [code...]), ...]"""
    rows = conn.execute(
        "SELECT id, code, pick_date, substr(created_at,1,16) AS batch "
        "FROM pick_tracking WHERE session_type=? ORDER BY created_at, id",
        (SESSION,),
    ).fetchall()
    batches: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for rid, code, pick_date, batch in rows:
        batches[(pick_date, batch)].append((rid, code))

    bad = []
    for (pick_date, batch), items in sorted(batches.items()):
        codes = [c for _, c in items]
        if _is_degenerate(codes):
            bad.append((pick_date, batch, [i for i, _ in items], codes))
    return bad


def rebuild_summary(conn) -> tuple[int, int]:
    """按原算法重放 post_market 明细，重建 pick_summary 并回填明细周期字段。"""
    conn.execute("DELETE FROM pick_summary WHERE session_type=?", (SESSION,))

    rows = conn.execute(
        "SELECT id, code, name, pick_date FROM pick_tracking "
        "WHERE session_type=? ORDER BY pick_date, id",
        (SESSION,),
    ).fetchall()

    # state[code] = dict(cumulative, cycle_id, cycle_start, cycle_end,
    #                    cycle_hits, total_cycles, first_pick_date)
    state: dict[str, dict] = {}
    seen_day: set[tuple[str, str]] = set()
    replayed = 0
    deduped = 0

    for rid, code, name, pick_date in rows:
        key = (code, pick_date)
        if key in seen_day:
            # 同日重复（历史遗留），原算法不会二次计数
            deduped += 1
            continue
        seen_day.add(key)

        end_of = (datetime.strptime(pick_date, "%Y-%m-%d")
                  + timedelta(days=CYCLE_CALENDAR_DAYS)).strftime("%Y-%m-%d")
        st = state.get(code)
        if st is None:
            st = {
                "cumulative": 1,
                "cycle_id": pick_date,
                "cycle_start": pick_date,
                "cycle_end": end_of,
                "cycle_hits": 1,
                "total_cycles": 1,
                "first_pick_date": pick_date,
                "name": name,
            }
            is_cycle_start = 1
            state[code] = st
        else:
            st["cumulative"] += 1
            if st["cycle_end"] and pick_date <= st["cycle_end"]:
                # 周期内: 延期 + 累加
                st["cycle_end"] = end_of
                st["cycle_hits"] += 1
                is_cycle_start = 0
            else:
                st["cycle_id"] = pick_date
                st["cycle_start"] = pick_date
                st["cycle_end"] = end_of
                st["cycle_hits"] = 1
                st["total_cycles"] += 1
                is_cycle_start = 1
            if name:
                st["name"] = name

        conn.execute(
            "UPDATE pick_tracking SET cycle_id=?, cycle_start=?, cycle_end=?, "
            "cycle_hits=?, cumulative=?, is_cycle_start=? WHERE id=?",
            (st["cycle_id"], st["cycle_start"], st["cycle_end"],
             st["cycle_hits"], st["cumulative"], is_cycle_start, rid),
        )
        st["last_pick_date"] = pick_date
        replayed += 1

    for code, st in state.items():
        conn.execute(
            "INSERT INTO pick_summary "
            "(code, name, session_type, cumulative_hits, active_cycle_id, "
            " active_cycle_hits, active_cycle_start, active_cycle_end, "
            " total_cycles, last_pick_date, first_pick_date) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (code, st["name"], SESSION, st["cumulative"], st["cycle_id"],
             st["cycle_hits"], st["cycle_start"], st["cycle_end"],
             st["total_cycles"], st["last_pick_date"], st["first_pick_date"]),
        )
    return replayed, deduped


def snapshot(conn, label: str) -> None:
    n_track = conn.execute(
        "SELECT COUNT(*) FROM pick_tracking WHERE session_type=?", (SESSION,)
    ).fetchone()[0]
    n_sum = conn.execute(
        "SELECT COUNT(*) FROM pick_summary WHERE session_type=?", (SESSION,)
    ).fetchone()[0]
    tot = conn.execute(
        "SELECT COALESCE(SUM(cumulative_hits),0) FROM pick_summary WHERE session_type=?",
        (SESSION,),
    ).fetchone()[0]
    print(f"\n--- {label} ---")
    print(f"  pick_tracking(post_market) 明细: {n_track}")
    print(f"  pick_summary(post_market)  股票: {n_sum}, 累计命中: {tot}")
    print("  Top8 累计命中:")
    for r in conn.execute(
        "SELECT code, name, cumulative_hits, last_pick_date FROM pick_summary "
        "WHERE session_type=? ORDER BY cumulative_hits DESC, code LIMIT 8",
        (SESSION,),
    ):
        print(f"    {r[0]}  {str(r[1] or '-'):10s} 命中={r[2]:2d}  最近={r[3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写库（默认 dry-run）")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    snapshot(conn, "修复前")

    bad = find_degenerate_batches(conn)
    ids: list[int] = []
    print("\n--- 退化批次判定 ---")
    if not bad:
        print("  未发现退化批次，无需清理。")
    for pick_date, batch, row_ids, codes in bad:
        ids.extend(row_ids)
        print(f"  {pick_date}  批次 {batch}  n={len(row_ids)}  "
              f"代码 {min(codes)}~{max(codes)}")
    print(f"  合计待删除: {len(ids)} 行")

    if not args.apply:
        print("\n[DRY-RUN] 未写库。确认无误后加 --apply 执行。")
        return 0

    if ids:
        conn.executemany("DELETE FROM pick_tracking WHERE id=?",
                         [(i,) for i in ids])
        print(f"\n已删除 {len(ids)} 行退化明细。")

    replayed, deduped = rebuild_summary(conn)
    conn.commit()
    print(f"已重放 {replayed} 条存活明细重建汇总（同日去重跳过 {deduped} 条）。")

    snapshot(conn, "修复后")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
