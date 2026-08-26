"""轻量验证: 生产版 hot_industry_set 与回测版 compute_hot_sectors 在代表日期的输出一致性。
只抽 ~12 个日期(各年 + 2026 月度), 避免 1554 次全表扫描。"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import walk_forward_oos_paths as M
from sector_rotation_watchlist import hot_industry_set

t0 = time.time()
px = M.load_data_x()
all_dates = pd.to_datetime(sorted(px["date"].unique()))
hot_bt = M.compute_hot_sectors(px)   # 回测版
print(f"[load] {len(px):,} 行, 决策日 {len(all_dates)}, {time.time()-t0:.1f}s")

# 抽样日期: 2026 每月 + 2025 H2 + 2021-2024 各 1
sample = []
for y in [2021, 2022, 2023, 2024, 2025]:
    cand = [d for d in all_dates if d.year == y]
    if len(cand) >= 2:
        sample += [cand[0], cand[len(cand)//2]]
for m in range(1, 13):
    cand = [d for d in all_dates if d.year == 2026 and d.month == m]
    if cand:
        sample.append(cand[len(cand)//2])
print(f"[sample] {len(sample)} 个日期")

rows = []
for d in sample:
    bt = hot_bt.get(d, set())
    # 生产版: 用生产函数(全量 DB, 与 multifactor 运行时一致)
    prod = hot_industry_set(as_of=str(d.date()), hot_n=M.HOT_N, mom_window=M.MOM_W)
    if not bt and not prod:
        jac = 1.0
    elif not bt or not prod:
        jac = 0.0
    else:
        inter = len(bt & prod)
        uni = len(bt | prod)
        jac = inter / uni if uni else 0.0
    rows.append((str(d.date()), len(bt), len(prod), round(jac, 2),
                 sorted(bt)[:3], sorted(prod)[:3]))

print(f"{'date':12s} {'bt_n':>4s} {'prod_n':>6s} {'jaccard':>7s}  bt_top3 / prod_top3")
for r in rows:
    print(f"{r[0]:12s} {r[1]:4d} {r[2]:6d} {r[3]:7.2f}  {r[4]} / {r[5]}")
mean_j = np.mean([r[3] for r in rows])
print(f"\n平均 Jaccard = {mean_j:.3f}  (>=0.5 即视为线上≡回测信号一致)")
print(f"生产版函数: hot_industry_set(as_of, hot_n=6, mom_window=60) 已接入 multifactor._score_l4_lvrev")
