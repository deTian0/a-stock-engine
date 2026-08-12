"""
alpha_research.py — A股因子「真 alpha」诊断（与持有期机制解耦）

目标: 量化每个因子本身的预测力，剥离 MIN_HOLD / 止损等"持有期机制"带来的账面转正。
方法:
  1. 时点正确性(PIT): 价格用 T 日及之前；基本面用 ann_date<=T 的最新财报(searchsorted backward)。
  2. 因子值: 在截面 T 上计算（动量/相对强度/趋势/低波/价值/质量，方向已统一为"越大越好"）。
  3. 前瞻收益: fwd_h = close(T+h)/close(T)-1（h=5/20/60 交易日）。
  4. Rank-IC: 每日截面内 factor 排名 与 fwd 排名 的 Spearman 相关 -> 均值 IC / ICIR / t 值。
  5. 多空组合(LS): 每 h 交易日调仓，多头部 decile、空尾部 decile、等权、持有 h 天 ->
     美元中性、无市场暴露，是"因子本身能否赚钱"的最干净度量。年化 = 均值 * (252/h)，
     Sharpe = 均值/标准差 * sqrt(252/h)，胜率 = 正收益调仓占比。
  6. 对比: 当前加权合成(factor_engine DEFAULT_WEIGHTS) 的 LS 收益 vs 仅取 IC 显著为正的
     因子的等权合成，看当前权重是否真的捕获了 alpha。

产物: briefs/alpha_research.csv + briefs/alpha_research.html
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB = Path(__file__).parent / "data_cache" / "market.db"
OUT = Path(__file__).parent / "briefs"
OUT.mkdir(parents=True, exist_ok=True)

LIQ = 50_000_000      # 日成交额下限(5 千万)，剔除微盘/僵尸股，保证 IC/LS 有意义
START = "2020-01-01"


def load_price():
    con = sqlite3.connect(str(DB))
    df = pd.read_sql_query(
        "SELECT code, date, close, pct_chg, amount FROM daily_price WHERE date>=?",
        con, params=(START,))
    con.close()
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def add_price_factors(df):
    """分组计算动量/均线/相对强度/趋势/波动（与 local_backtest 同义）。"""
    g = df.groupby("code", sort=False)
    df["chg_10d"] = g["close"].transform(lambda s: s / s.shift(10) - 1)
    df["chg_25d"] = g["close"].transform(lambda s: s / s.shift(25) - 1)
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["ma60"] = g["close"].transform(lambda s: s.rolling(60).mean())
    df["rs20"] = np.where(df["ma20"] > 0, df["close"] / df["ma20"] - 1, np.nan)
    df["trend_up"] = (df["ma20"] > df["ma60"]).astype(float)  # 1 多头 / 0 空头
    df["vol20"] = g["pct_chg"].transform(lambda s: s.rolling(20).std())
    for h in (5, 20, 60):
        df[f"fwd{h}"] = g["close"].transform(lambda s: s.shift(-h) / s - 1)
    return df


def merge_pit(df):
    """PIT 合并基本面: 每只股票在 T 日取 ann_date<=T 的最新财报(不前瞻, 无 suffix 错配)。"""
    con = sqlite3.connect(str(DB))
    fin = pd.read_sql_query(
        "SELECT code, ann_date, roe, gross_margin, debt_ratio, "
        "revenue_growth, profit_growth, eps_ttm, bps FROM fundamentals_pit", con)
    con.close()
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin["ann_dt"] = pd.to_datetime(fin["ann_date"], errors="coerce")
    fin = fin.dropna(subset=["ann_dt"]).sort_values(["code", "ann_dt"])

    fcols = ["roe", "gross_margin", "debt_ratio", "revenue_growth",
             "profit_growth", "eps_ttm", "bps"]
    df = df.copy()
    df["_c"] = (df["code"].astype(str)
                .str.replace(r"\.(SZ|SH|BJ)$", "", regex=True).str.zfill(6))
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("_c").reset_index(drop=True)

    ann_map, val_map = {}, {}
    for code, g in fin.groupby("code"):
        ann_map[code] = g["ann_dt"].values
        val_map[code] = g[fcols].values

    out = np.full((len(df), len(fcols)), np.nan)
    codes = df["_c"].values
    dts = df["_dt"].values
    is_new = np.ones(len(codes), bool)
    is_new[1:] = codes[1:] != codes[:-1]
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], len(codes))
    for s, e in zip(starts, ends):
        code = codes[s]
        if code not in ann_map:
            continue
        ann = ann_map[code]
        idx = np.searchsorted(ann, dts[s:e], side="right") - 1   # 最新 <=T
        valid = idx >= 0
        idxc = np.clip(idx, 0, len(ann) - 1)
        rows = np.arange(s, e)[valid]
        out[rows] = val_map[code][idxc[valid]]     # 早于首份财报保持 NaN(不前瞻)

    for j, c in enumerate(fcols):
        df[c] = out[:, j]
    df["earn_yield"] = np.where(df["close"] > 0, df["eps_ttm"] / df["close"], np.nan)
    return df.drop(columns=["_c", "_dt"])


def ic_from_ranks(rf, ry, dates):
    """rf, ry 为已排名的 Series（同 index），dates 为对应日期 Series。返回逐日 IC。"""
    tmp = pd.DataFrame({"rf": rf, "ry": ry, "d": dates}).dropna()
    if len(tmp) < 100:
        return pd.Series(dtype=float)
    g = tmp.groupby("d")
    a = tmp["rf"] - g["rf"].transform("mean")
    b = tmp["ry"] - g["ry"].transform("mean")
    num = (a * b).groupby(tmp["d"]).sum()
    den = np.sqrt((a * a).groupby(tmp["d"]).sum() * (b * b).groupby(tmp["d"]).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def ls_from_decile(dec, fwd, dates, h):
    """dec: 0..9 的 decile Series（同 index）；返回逐次调仓多空收益序列。"""
    need = pd.DataFrame({"dec": dec, "fwd": fwd, "d": dates}).dropna()
    if len(need) < 500:
        return pd.Series(dtype=float)
    du = sorted(need["d"].unique())
    reb = set(du[i] for i in range(0, len(du), h))
    sub = need[need["d"].isin(reb)]
    if len(sub) < 50:
        return pd.Series(dtype=float)
    long_m = sub[sub["dec"] == 9].groupby("d")["fwd"].mean()
    short_m = sub[sub["dec"] == 0].groupby("d")["fwd"].mean()
    return (long_m - short_m).dropna()


def ls_stats(s, h):
    if len(s) == 0:
        return (np.nan, np.nan, np.nan)
    ann = s.mean() * (252.0 / h)
    sr = (s.mean() / s.std()) * np.sqrt(252.0 / h) if s.std() else 0
    wr = (s > 0).mean() * 100
    return round(ann * 100, 2), round(sr, 2), round(wr, 1)


def main():
    print("加载价格数据 ...")
    df = load_price()
    print(f"  价格行: {len(df):,}  股票: {df['code'].nunique():,}")
    df = add_price_factors(df)
    print("合并 PIT 基本面 ...")
    df = merge_pit(df)
    df = df[(df["amount"] >= LIQ) & (df["close"] > 0)].copy()
    print(f"  过滤后(日成交额>={LIQ/1e6:.0f}M): {len(df):,} 行")

    factors = {
        "chg_10d":       ("动量10日", +1),
        "chg_25d":       ("动量25日", +1),
        "rs20":          ("相对强度RS", +1),
        "trend_up":      ("趋势MA20>MA60", +1),
        "vol20":         ("低波动(反向)", -1),
        "earn_yield":    ("盈利收益率(价值)", +1),
        "roe":           ("ROE(质量)", +1),
        "gross_margin":  ("毛利率(质量)", +1),
        "debt_ratio":    ("负债率(反向)", -1),
        "revenue_growth":("营收增速(质量)", +1),
        "profit_growth": ("利润增速(质量)", +1),
    }
    scols = {}
    for f, (_, d) in factors.items():
        s = (-df[f]) if (d < 0 and f in df.columns) else df[f]
        df[f + "_s"] = s
        scols[f] = f + "_s"

    horizons = [5, 20, 60]
    fwd_cols = {h: f"fwd{h}" for h in horizons}

    # 一次性算排名(每 horizon 对所有因子+前瞻收益各一次 groupby)，避免重复 rank
    print("计算截面排名 ...")
    fac_rank = {h: df.groupby("date")[[scols[f] for f in factors]].rank() for h in horizons}
    fwd_rank = {h: df.groupby("date")[fwd_cols[h]].rank() for h in horizons}
    fac_pct = {h: df.groupby("date")[[scols[f] for f in factors]].rank(pct=True)
               for h in (20, 60)}

    rows = []
    ls_cache = {}   # (factor, h) -> Series
    for f, (label, _) in factors.items():
        fs = scols[f]
        rec = {"factor": f, "label": label}
        for h in horizons:
            ic = ic_from_ranks(fac_rank[h][fs], fwd_rank[h], df["date"])
            if len(ic):
                rec[f"IC{h}"] = round(ic.mean(), 4)
                rec[f"ICIR{h}"] = round(ic.mean() / ic.std(), 3) if ic.std() else 0
                rec[f"t{h}"] = round(ic.mean() / (ic.std() / np.sqrt(len(ic))), 2) if ic.std() else 0
            else:
                rec[f"IC{h}"] = rec[f"ICIR{h}"] = rec[f"t{h}"] = np.nan
        for h, key in ((20, "LS20"), (60, "LS60")):
            dec = np.floor(fac_pct[h][fs] * 10).clip(0, 9)
            s = ls_from_decile(dec, df[fwd_cols[h]], df["date"], h)
            ls_cache[(f, h)] = s
            ann, sr, wr = ls_stats(s, h)
            rec[f"{key}_ann%"] = ann
            rec[f"{key}_SR"] = sr
            rec[f"{key}_WR%"] = wr
        rows.append(rec)

    res = pd.DataFrame(rows)
    print("\n=== 因子 Rank-IC 与多空收益 ===")
    cols_show = ["label", "IC5", "IC20", "IC60", "ICIR20", "t20",
                 "LS20_ann%", "LS20_SR", "LS20_WR%", "LS60_ann%"]
    print(res[cols_show].to_string(index=False))
    res.to_csv(OUT / "alpha_research.csv", index=False)
    print(f"\n已保存: {OUT / 'alpha_research.csv'}")

    # IC20 显著为正的因子 -> 等权合成
    sub = res.set_index("factor")
    sig = [f for f in factors
           if not np.isnan(sub.loc[f, "IC20"]) and sub.loc[f, "IC20"] > 0 and sub.loc[f, "t20"] > 1.5]
    print(f"\nIC20 显著为正(>0 且 t>1.5)因子: {sig}")
    s_sig20 = s_sig60 = None
    if sig:
        comp = df[[scols[s] for s in sig]].fillna(0).mean(axis=1)
        dec = np.floor(comp.groupby(df["date"]).rank(pct=True) * 10).clip(0, 9)
        s_sig20 = ls_from_decile(dec, df["fwd20"], df["date"], 20)
        s_sig60 = ls_from_decile(dec, df["fwd60"], df["date"], 60)
        a, sr, w = ls_stats(s_sig20, 20)
        print(f"  IC显著等权合成 LS20 年化: {a:+.2f}%  Sharpe: {sr:.2f}  胜率: {w:.1f}%")
        a, sr, w = ls_stats(s_sig60, 60)
        print(f"  IC显著等权合成 LS60 年化: {a:+.2f}%  Sharpe: {sr:.2f}  胜率: {w:.1f}%")

    # 当前 DEFAULT_WEIGHTS 合成
    cur_w = {"chg_10d": 0.06, "chg_25d": 0.05, "rs20": 0.18, "trend_up": 0.10,
             "vol20": -0.10, "earn_yield": 0.10, "roe": 0.03, "gross_margin": 0.02,
             "debt_ratio": -0.02, "revenue_growth": 0.015, "profit_growth": 0.015}
    avail = {k: v for k, v in cur_w.items() if scols.get(k) in df.columns}
    wsum = sum(abs(v) for v in avail.values()) or 1
    comp_cur = sum(df[scols[k]].fillna(0) * (v / wsum) for k, v in avail.items())
    dec_cur = np.floor(comp_cur.groupby(df["date"]).rank(pct=True) * 10).clip(0, 9)
    s_cur20 = ls_from_decile(dec_cur, df["fwd20"], df["date"], 20)
    s_cur60 = ls_from_decile(dec_cur, df["fwd60"], df["date"], 60)
    a, sr, w = ls_stats(s_cur20, 20)
    print(f"  当前权重合成   LS20 年化: {a:+.2f}%  Sharpe: {sr:.2f}  胜率: {w:.1f}%")
    a, sr, w = ls_stats(s_cur60, 60)
    print(f"  当前权重合成   LS60 年化: {a:+.2f}%  Sharpe: {sr:.2f}  胜率: {w:.1f}%")

    build_html(res, sig, s_sig20, s_cur20)
    return res


def build_html(res, sig, s_sig, s_cur):
    r = res.copy()
    labels = r["label"].tolist()
    ic20 = r["IC20"].fillna(0).tolist()
    ls20 = r["LS20_ann%"].fillna(0).tolist()

    def cum(ret_series):
        if ret_series is None or len(ret_series) == 0:
            return []
        return [round(x, 4) for x in (1 + ret_series).cumprod().tolist()]

    cum_sig = cum(s_sig)
    cum_cur = cum(s_cur)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>因子真 Alpha 诊断</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#0f1419;color:#e6e6e6;margin:0;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}}.sub{{color:#8b98a5;font-size:13px;margin-bottom:18px}}
.card{{background:#1a2230;border:1px solid #2a3548;border-radius:10px;padding:18px;margin-bottom:18px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #2a3548;padding:6px 8px;text-align:center}}
th{{background:#22304a;color:#cfe}}tr:nth-child(even){{background:#161d29}}
.pos{{color:#ff5b5b}}.neg{{color:#26a69a}}
.grid{{display:flex;gap:18px;flex-wrap:wrap}}.grid>.card{{flex:1;min-width:380px}}
.note{{color:#8b98a5;font-size:12px;line-height:1.6}}</style></head><body>
<h1>A股因子「真 Alpha」诊断</h1>
<div class="sub">PIT 正确 · 剥离持有期机制 · 美元中性多空(decile 1 vs 9) · 窗口 2020-01 起 · 日成交额≥50M</div>
<div class="card"><div class="note">
<b>读法</b>：Rank-IC 衡量因子排序对 h 日后收益的解释力（|IC|&gt;0.03 弱、&gt;0.05 中、&gt;0.08 强；ICIR&gt;0.5 稳健）。
多空年化 = 头部 decile − 尾部 decile 等权、每 h 日调仓持有 h 天的年化收益，<b>无市场暴露</b>，是因子本身能否赚钱的最干净度量。
当前策略的"翻正"主要来自 MIN_HOLD=30 的持有期机制（给盈利仓位时间跑、压交易成本），而非因子排序本身有强 alpha。
</div></div>
<div class="grid">
<div class="card"><canvas id="ic" height="320"></canvas></div>
<div class="card"><canvas id="ls" height="320"></canvas></div>
</div>
<div class="card"><table>
<tr><th>因子</th><th>IC5</th><th>IC20</th><th>IC60</th><th>ICIR20</th><th>t20</th><th>LS20年化%</th><th>LS20夏普</th><th>LS20胜率%</th><th>LS60年化%</th></tr>
"""
    for _, row in r.iterrows():
        def fmt(v, p=2):
            return "-" if pd.isna(v) else f"{v:.{p}f}"
        ic20c = "pos" if (not pd.isna(row["IC20"]) and row["IC20"] > 0) else "neg"
        ls20c = "pos" if (not pd.isna(row["LS20_ann%"]) and row["LS20_ann%"] > 0) else "neg"
        html += (f"<tr><td style='text-align:left'>{row['label']}</td>"
                 f"<td>{fmt(row['IC5'])}</td>"
                 f"<td class='{ic20c}'>{fmt(row['IC20'])}</td>"
                 f"<td>{fmt(row['IC60'])}</td>"
                 f"<td>{fmt(row['ICIR20'])}</td>"
                 f"<td>{fmt(row['t20'])}</td>"
                 f"<td class='{ls20c}'>{fmt(row['LS20_ann%'])}</td>"
                 f"<td>{fmt(row['LS20_SR'])}</td>"
                 f"<td>{fmt(row['LS20_WR%'],1)}</td>"
                 f"<td>{fmt(row['LS60_ann%'])}</td></tr>\n")
    html += "</table></div>"

    if cum_sig or cum_cur:
        html += '<div class="card"><canvas id="cum" height="300"></canvas>'
        html += '<div class="note">多空累计净值：蓝=IC显著因子等权合成，橙=当前权重合成（均为 decile 多空，每20日调仓）。</div></div>'

    html += """
<script>
const labels = """ + str(labels) + """;
const ic20 = """ + str(ic20) + """;
const ls20 = """ + str(ls20) + """;
new Chart(document.getElementById('ic'),{type:'bar',data:{labels,datasets:[{label:'IC20',data:ic20,backgroundColor:ic20.map(v=>v>=0?'#ff5b5b':'#26a69a')}]},options:{plugins:{title:{display:true,text:'因子 Rank-IC (20日)',color:'#cfe'},legend:{display:false}},scales:{y:{grid:{color:'#2a3548'},ticks:{color:'#8b98a5'}},x:{grid:{color:'#2a3548'},ticks:{color:'#8b98a5',maxRotation:60,minRotation:45}}}}});
new Chart(document.getElementById('ls'),{type:'bar',data:{labels,datasets:[{label:'LS20年化%',data:ls20,backgroundColor:ls20.map(v=>v>=0?'#ff5b5b':'#26a69a')}]},options:{plugins:{title:{display:true,text:'多空年化收益% (20日)',color:'#cfe'},legend:{display:false}},scales:{y:{grid:{color:'#2a3548'},ticks:{color:'#8b98a5'}},x:{grid:{color:'#2a3548'},ticks:{color:'#8b98a5',maxRotation:60,minRotation:45}}}}});
"""
    if cum_sig or cum_cur:
        html += "const cumSig=" + str(cum_sig) + ";const cumCur=" + str(cum_cur) + ";" + """
const ds=[];
if(cumSig.length)ds.push({label:'IC显著等权合成',data:cumSig,borderColor:'#4da3ff',fill:false});
if(cumCur.length)ds.push({label:'当前权重合成',data:cumCur,borderColor:'#ffa726',fill:false});
new Chart(document.getElementById('cum'),{type:'line',data:{labels:ds[0].data.map((_,i)=>i),datasets:ds},options:{plugins:{title:{display:true,text:'多空累计净值 (每20日调仓)',color:'#cfe'}},scales:{y:{grid:{color:'#2a3548'},ticks:{color:'#8b98a5'}},x:{grid:{color:'#2a3548'},ticks:{color:'#8b98a5'}}}}});
"""
    html += "</script></body></html>"
    p = OUT / "alpha_research.html"
    p.write_text(html, encoding="utf-8")
    print(f"已保存: {p}")


if __name__ == "__main__":
    main()
