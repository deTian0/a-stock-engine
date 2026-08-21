"""
compute_ic.py — A股 lvrev 四因子截面 Rank-IC 重算 (2020-2026)
数据源: data_cache/market.db
  - 低波(low_vol) / 反转(reversal): daily_price 价格序列推导
  - 质量(roe) / 成长(profit_growth): fundamentals_pit (PIT, 时点正确, 杜绝前视)
输出: empirical/ic_summary.txt + empirical/ic_results.csv

Rank-IC = 每个时点 t, 因子截面排名 与 未来 N 日收益截面排名 的 Spearman 相关;
        跨所有 t 取均值 = mean_IC; ICIR = mean_IC / std(IC); t-stat = mean/(std/sqrt(n)).
"""
import sqlite3, os, time
import numpy as np
import pandas as pd
from pathlib import Path

DB = r"D:\workspace\github\a-stock-engine\data_cache\market.db"
OUTDIR = r"D:\workspace\github\a-stock-engine\empirical"
os.makedirs(OUTDIR, exist_ok=True)

t0 = time.time()
c = sqlite3.connect(DB)

# ---------- 价格矩阵 ----------
print("[load] daily_price ...", flush=True)
dp = pd.read_sql("SELECT code, date, close FROM daily_price WHERE date>='2020-01-01'", c)
dp['date'] = pd.to_datetime(dp['date'])
dp = dp.sort_values(['code', 'date'])
close = dp.pivot(index='date', columns='code', values='close').sort_index()
codes = list(close.columns)
dates = close.index
print(f"[load] close matrix {close.shape}, codes={len(codes)}, dates={len(dates)} ({time.time()-t0:.1f}s)", flush=True)

ret = close.pct_change(1)
low_vol = -ret.rolling(20).std()                       # 低波: 负20日实现波动
reversal = -(close / close.shift(20) - 1)              # 反转: 负近20日累积收益(超跌)
fwd20 = close.shift(-20) / close - 1
fwd60 = close.shift(-60) / close - 1


def rank_ic_series(factor: pd.DataFrame, fwd: pd.DataFrame):
    """逐时点截面 Rank-IC, 返回 Series(按日期)。"""
    ics = []
    for t in factor.index:
        f = factor.loc[t]
        g = fwd.loc[t]
        m = f.notna() & g.notna()
        if m.sum() < 50:
            continue
        try:
            ic = f[m].rank().corr(g[m].rank())
            if pd.notna(ic):
                ics.append(ic)
        except Exception:
            pass
    return pd.Series(ics).dropna()


def summarize(name, ics: pd.Series):
    n = len(ics)
    mean = ics.mean()
    std = ics.std()
    icir = mean / std if std and std > 0 else 0.0
    tstat = mean / (std / np.sqrt(n)) if std and std > 0 else 0.0
    pos = (ics > 0).mean() * 100
    return {"factor": name, "n_dates": n, "mean_IC": round(mean, 4),
            "std_IC": round(std, 4), "ICIR": round(icir, 3),
            "t_stat": round(tstat, 2), "pct_IC_positive": round(pos, 1)}


rows = []
# 价格因子
for fname, fac in [("low_vol(20d实现波动取负)", low_vol), ("reversal(近20日超跌取负)", reversal)]:
    for hname, fwd in [("fwd20", fwd20), ("fwd60", fwd60)]:
        ics = rank_ic_series(fac, fwd)
        s = summarize(f"{fname} | {hname}", ics)
        rows.append(s)
        print(f"[ic] {s}", flush=True)

# ---------- 质量/成长: PIT 基本面, 季度频率 ----------
print("[load] fundamentals_pit ...", flush=True)
# daily_price.code 带交易所后缀(000001.SZ), fundamentals_pit.code 无后缀(000001) -> 需统一
strip_suffix = lambda s: s.split('.')[0] if '.' in str(s) else s
code_map = {strip_suffix(x): x for x in close.columns}   # 无后缀6位 -> 带后缀
pit = pd.read_sql("SELECT code, ann_date, roe, profit_growth FROM fundamentals_pit", c)
pit['code_full'] = pit['code'].map(code_map)            # 对齐 daily_price 代码格式
pit = pit.dropna(subset=['code_full'])
pit['ann_date'] = pd.to_datetime(pit['ann_date'], format='%Y%m%d', errors='coerce')
pit = pit.dropna(subset=['ann_date']).sort_values('ann_date')
print(f"[load] pit matched {len(pit)} rows across {pit['code_full'].nunique()} codes", flush=True)

# 季度末交易日
qdates = pd.date_range('2020-03-31', close.index[-1], freq='QE')
qdates = [d for d in qdates if d in close.index]
print(f"[ic] quarterly rebalance dates: {len(qdates)}", flush=True)


def ic_quarterly(col, horizon=60):
    fwd = close.shift(-horizon) / close - 1
    ics = []
    for t in qdates:
        sub = pit[pit['ann_date'] <= t]
        if len(sub) == 0:
            continue
        last = sub.sort_values('ann_date').groupby('code_full').last()
        avail = [x for x in last.index if x in close.columns]
        f = last.loc[avail, col]
        g = fwd.loc[t, avail]
        m = f.notna() & g.notna()
        if m.sum() < 50:
            continue
        try:
            ic = f[m].rank().corr(g[m].rank())
            if pd.notna(ic):
                ics.append(ic)
        except Exception:
            pass
    return pd.Series(ics).dropna()


for fname, col in [("quality(ROE, PIT)", "roe"), ("growth(利润增速, PIT)", "profit_growth")]:
    ics = ic_quarterly(col, horizon=60)
    s = summarize(f"{fname} | fwd60(季度)", ics)
    rows.append(s)
    print(f"[ic] {s}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "ic_results.csv"), index=False)
with open(os.path.join(OUTDIR, "ic_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write("A股 lvrev 四因子截面 Rank-IC (2020-2026)\n")
    fh.write(f"样本: daily_price {close.shape[0]}交易日 x {len(codes)}只; PIT基本面 {len(pit)}条\n")
    fh.write(f"耗时: {time.time()-t0:.1f}s\n\n")
    fh.write(df.to_string(index=False))
print(f"\n[DONE] {time.time()-t0:.1f}s -> empirical/ic_results.csv", flush=True)
