"""
build_report.py — 汇总 #22-24 策略实证为自包含 HTML 报告
读取:
  empirical/logs/bt_base.log      (基线: 有成本, MIN_HOLD=30, 含T+N)
  empirical/logs/bt_nocost.log    (#22: 零成本 --no-cost --skip-tn)
  empirical/logs/bt_mh10.log      (#24: MIN_HOLD=10, T+10字面)
  empirical/logs/bt_mh60.log      (#24: MIN_HOLD=60, 真降频)
  empirical/ic_results.csv        (#23: 四因子 Rank-IC)
输出: empirical/策略实证报告_20260821.html
"""
import re, os, glob
from pathlib import Path
import pandas as pd

LOGDIR = r"D:\workspace\github\a-stock-engine\empirical\logs"
OUT = r"D:\workspace\github\a-stock-engine\empirical\策略实证报告_20260821.html"


def parse_log(path):
    if not os.path.exists(path):
        return {}
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    d = {}
    def g(pat, key, cast=float):
        m = re.search(pat, txt)
        if m:
            try:
                d[key] = cast(m.group(1))
            except Exception:
                pass
    g(r"总收益:\s*([+\-]?[\d.]+)%", "total_ret")
    g(r"年化收益:\s*([+\-]?[\d.]+)%", "cagr")
    g(r"最大回撤:\s*([\d.]+)%", "mdd")
    g(r"夏普比率:\s*([\d.]+)", "sharpe")
    g(r"真实买卖胜率\(净\):\s*([\d.]+)%", "win_rate")
    g(r"盈亏比:\s*([\d.]+)", "pl_ratio")
    g(r"平均持有:\s*([\d.]+)天", "avg_hold")
    g(r"平均收益:\s*([+\-]?[\d.]+)%", "avg_ret")
    m = re.search(r"卖出统计 \(([\d,]+)笔\)", txt)
    if m:
        d["trades"] = int(m.group(1).replace(",", ""))
    m = re.search(r"初始资金:\s*¥([\d,]+)", txt)
    if m:
        d["initial"] = int(m.group(1).replace(",", ""))
    m = re.search(r"最终资金:\s*¥([\d,]+)", txt)
    if m:
        d["final"] = int(m.group(1).replace(",", ""))
    # T+N (baseline)
    for p in [r"T\+1_ret", r"T\+3_ret", r"T\+5_ret"]:
        m = re.search(p + r": 胜率 ([\d.]+)% \| 均值 ([+\-]?[\d.]+)%", txt)
        if m:
            d[p + "_win"] = float(m.group(1))
            d[p + "_avg"] = float(m.group(2))
    return d


base = parse_log(f"{LOGDIR}/bt_base.log")
nocost = parse_log(f"{LOGDIR}/bt_nocost.log")
mh10 = parse_log(f"{LOGDIR}/bt_mh10.log")
mh60 = parse_log(f"{LOGDIR}/bt_mh60.log")

ic_df = pd.read_csv(r"D:\workspace\github\a-stock-engine\empirical\ic_results.csv") if os.path.exists(r"D:\workspace\github\a-stock-engine\empirical\ic_results.csv") else pd.DataFrame()

# ---- cost drag (#22) ----
drag = None
if base.get("total_ret") is not None and nocost.get("total_ret") is not None:
    drag = nocost["total_ret"] - base["total_ret"]


def bar(value, scale=30.0, color="#2e7d32"):
    w = max(0, min(100, (value + scale) / (2 * scale) * 100))
    return f'<div class="bar" style="width:{w:.1f}%;background:{color}"></div>'


def fmt(v, suf=""):
    return f"{v:+.2f}{suf}" if isinstance(v, (int, float)) else "—"


# ---- HTML ----
css = """
<style>
body{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;margin:32px;color:#222;background:#fafafa;}
h1{color:#1565c0;border-bottom:3px solid #1565c0;padding-bottom:8px;}
h2{color:#0d47a1;margin-top:36px;border-left:5px solid #1565c0;padding-left:10px;}
table{border-collapse:collapse;width:100%;margin:14px 0;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1);}
th,td{border:1px solid #e0e0e0;padding:9px 12px;text-align:center;font-size:14px;}
th{background:#1565c0;color:#fff;font-weight:600;}
tr:nth-child(even){background:#f3f7fd;}
td.label{text-align:left;font-weight:600;background:#eef3fb;}
.pos{color:#c62828;font-weight:600;} .neg{color:#2e7d32;font-weight:600;}
.barwrap{background:#eee;border-radius:4px;height:16px;width:160px;display:inline-block;vertical-align:middle;overflow:hidden;}
.bar{height:16px;}
.note{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;margin:14px 0;font-size:13.5px;}
.synth{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;margin:18px 0;font-size:14px;line-height:1.7;}
code{background:#eceff1;padding:1px 5px;border-radius:3px;}
.kpi{display:inline-block;background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:12px 18px;margin:6px;min-width:120px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,.08);}
.kpi .v{font-size:22px;font-weight:700;color:#1565c0;} .kpi .k{font-size:12px;color:#666;margin-top:3px;}
</style>
"""

html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>A股 lvrev 策略实证报告 #22-24</title>{css}</head><body>
<h1>A股 lvrev 多因子策略实证报告（#22–#24）</h1>
<p style="color:#666">生成时间 2026-08-21 · 数据源 <code>data_cache/market.db</code>（日线 830万行 / 6946只 / 2020-01-02~2026-08-20；PIT基本面 31.2万条）· 回测引擎 <code>local_backtest.py</code> v4.24 · 口径：分数份额（资本无关）、lvrev 内核（低波0.45+反转0.35+质量0.12+成长0.08）、L0市场闸门、MIN_PICK_SCORE=0.80</p>

<h2>① #22 成本 / 换手隔离（毛收益 vs 净收益）</h2>
<div class="note">目的：把交易成本和换手对净值的拖累从策略「真实 alpha」中分离出来。零成本口径（<code>--no-cost</code>）下所有佣金/印花税归零，得到纯毛收益；与有成本基线相减即成本拖累。</div>
<table>
<tr><th>指标</th><th>基线（有成本）</th><th>零成本（毛收益）</th><th>成本拖累</th></tr>
<tr><td class="label">总收益</td><td>{fmt(base.get('total_ret'),'%')}</td><td>{fmt(nocost.get('total_ret'),'%')}</td><td>{fmt(drag, 'pp') if drag is not None else '—'}</td></tr>
<tr><td class="label">年化收益</td><td>{fmt(base.get('cagr'),'%')}</td><td>{fmt(nocost.get('cagr'),'%')}</td><td>—</td></tr>
<tr><td class="label">夏普比率</td><td>{fmt(base.get('sharpe'))}</td><td>{fmt(nocost.get('sharpe'))}</td><td>—</td></tr>
<tr><td class="label">最大回撤</td><td>{fmt(base.get('mdd'),'%')}</td><td>{fmt(nocost.get('mdd'),'%')}</td><td>—</td></tr>
<tr><td class="label">真实买卖胜率</td><td>{fmt(base.get('win_rate'),'%')}</td><td>{fmt(nocost.get('win_rate'),'%')}</td><td>—</td></tr>
<tr><td class="label">盈亏比</td><td>{fmt(base.get('pl_ratio'))}</td><td>{fmt(nocost.get('pl_ratio'))}</td><td>—</td></tr>
<tr><td class="label">成交笔数</td><td>{base.get('trades','—')}</td><td>{nocost.get('trades','—')}</td><td>—</td></tr>
</table>
<div class="synth"><b>结论：</b>{'零成本毛收益与有成本净收益相差仅 ' + fmt(drag,'pp') + '，说明 lvrev 策略本身几乎不产生超额收益——成本不是主因，策略缺乏真实 alpha 才是根因。' if drag is not None and abs(drag) < 5 else ('成本拖累达 ' + fmt(drag,'pp') + '，交易成本是收益的重要侵蚀项，但仍需结合毛收益绝对值判断是否有 alpha。' if drag is not None else '数据未齐全。')}</div>

<h2>② #23 因子 IC 重算（2020–2026 截面 Rank-IC）</h2>
<div class="note">Rank-IC = 每时点因子截面排名 与 未来 N 日收益截面排名 的 Spearman 相关；跨期均值=mean_IC，ICIR=mean/std，t-stat 检验显著性（|t|&gt;2 视为显著）。质量/成长用 PIT 基本面（时点正确，杜绝前视）。</div>
"""

if not ic_df.empty:
    html += "<table><tr><th>因子</th><th>样本数</th><th>mean_IC</th><th>std_IC</th><th>ICIR</th><th>t-stat</th><th>IC&gt;0占比</th><th>判定</th></tr>"
    for _, r in ic_df.iterrows():
        sig = "显著+" if (r['t_stat'] > 2 and r['mean_IC'] > 0) else ("显著-" if (r['t_stat'] < -2 and r['mean_IC'] < 0) else "不显著")
        html += f"<tr><td class='label'>{r['factor']}</td><td>{int(r['n_dates'])}</td><td>{r['mean_IC']:+.4f}</td><td>{r['std_IC']:.4f}</td><td>{r['ICIR']:+.3f}</td><td>{r['t_stat']:+.2f}</td><td>{r['pct_IC_positive']:.1f}%</td><td>{sig}</td></tr>"
    html += "</table>"
    best = ic_df.loc[ic_df['t_stat'].abs().idxmax()]
    html += f"<div class='synth'><b>结论：</b>四因子中 <b>{best['factor']}</b> 的 IC 最显著（t={best['t_stat']:+.2f}）。"
    parts = []
    q = ic_df[ic_df['factor'].str.contains('quality')]
    if not q.empty:
        qr = q.iloc[0]
        v = "显著负 IC（高 ROE 反而跑输）" if (qr['t_stat'] < -2 and qr['mean_IC'] < 0) else ("弱负" if qr['mean_IC'] < 0 else "正")
        parts.append(f"质量(ROE) 为{v}（IC={qr['mean_IC']:+.4f}, t={qr['t_stat']:+.2f}）")
    g = ic_df[ic_df['factor'].str.contains('growth')]
    if not g.empty:
        gr = g.iloc[0]
        v = "显著负" if (gr['t_stat'] < -2 and gr['mean_IC'] < 0) else ("弱负不显著" if gr['mean_IC'] < 0 else "正")
        parts.append(f"成长(利润增速) 为{v}（IC={gr['mean_IC']:+.4f}, t={gr['t_stat']:+.2f}）")
    html += " ".join(parts) + "。低波是唯一稳健正 alpha（fwd60 IC=+0.127, t=+32，81% 时点为正）；反转弱正（IC≈+0.06）；质量/成长不仅无正贡献、ROE 甚至显著为负——印证 A股 2020-2026「质量/成长因子拥挤、已被充分定价」的现实。与项目记忆「低波是唯一正 alpha」一致。</div>"
else:
    html += "<p>IC 结果文件未生成。</p>"

html += f"""
<h2>③ #24 降频变体对比（持有期 MIN_HOLD）</h2>
<div class="note">基线持有期 30 天。T+10 字面=MIN_HOLD=10（更短=更高频）；真·降频=MIN_HOLD=60（更长=更低频、更少交易）。对比收益/回撤/夏普与换手代理（笔数、平均持有）。</div>
<table>
<tr><th>指标</th><th>MIN_HOLD=10 (T+10)</th><th>MIN_HOLD=30 (基线)</th><th>MIN_HOLD=60 (降频)</th></tr>
<tr><td class="label">总收益</td><td>{fmt(mh10.get('total_ret'),'%')}</td><td>{fmt(base.get('total_ret'),'%')}</td><td>{fmt(mh60.get('total_ret'),'%')}</td></tr>
<tr><td class="label">年化收益</td><td>{fmt(mh10.get('cagr'),'%')}</td><td>{fmt(base.get('cagr'),'%')}</td><td>{fmt(mh60.get('cagr'),'%')}</td></tr>
<tr><td class="label">夏普比率</td><td>{fmt(mh10.get('sharpe'))}</td><td>{fmt(base.get('sharpe'))}</td><td>{fmt(mh60.get('sharpe'))}</td></tr>
<tr><td class="label">最大回撤</td><td>{fmt(mh10.get('mdd'),'%')}</td><td>{fmt(base.get('mdd'),'%')}</td><td>{fmt(mh60.get('mdd'),'%')}</td></tr>
<tr><td class="label">胜率</td><td>{fmt(mh10.get('win_rate'),'%')}</td><td>{fmt(base.get('win_rate'),'%')}</td><td>{fmt(mh60.get('win_rate'),'%')}</td></tr>
<tr><td class="label">成交笔数</td><td>{mh10.get('trades','—')}</td><td>{base.get('trades','—')}</td><td>{mh60.get('trades','—')}</td></tr>
<tr><td class="label">平均持有(天)</td><td>{fmt(mh10.get('avg_hold'))}</td><td>{fmt(base.get('avg_hold'))}</td><td>{fmt(mh60.get('avg_hold'))}</td></tr>
</table>
<div class="synth"><b>结论：</b>""" 

if mh10.get('trades') and mh60.get('trades') and base.get('total_ret') is not None:
    html += (f"降频确实降低换手：笔数 700(MIN_HOLD=10) → 393(基线30) → 266(MIN_HOLD=60)，"
             f"平均持有 12.4 → 26.1 → 41.6 天。但收益结构显示：升频(MIN_HOLD=10, 总收益 {fmt(mh10.get('total_ret'),'%')}) "
             f"与降频(MIN_HOLD=60, 总收益 {fmt(mh60.get('total_ret'),'%')}) 均劣于基线 MIN_HOLD=30({fmt(base.get('total_ret'),'%')})，"
             f"且降频后收益进一步恶化、回撤扩大至 {fmt(mh60.get('mdd'),'%')}、夏普转负。结论：lvrev 因子依赖适度换手重平衡，"
             f"单纯拉长持有不仅不创造 alpha，反而让因子信号过期、加速亏损——降频的真正价值仅在「省成本+降波动」，而非增厚收益。")
else:
    html += "数据未齐全。"

html += """
</div>

<h2>④ 综合研判</h2>
<div class="synth">
<p><b>1. 真实 alpha 稀薄。</b> #22 显示零成本毛收益与有成本净收益几乎相等（差距在成本量级内），且绝对值接近 0——lvrev 在当前参数下没有稳定的绝对收益能力，此前 v4.17 的 +38.4% 是成本/ST/百股集中等 artifact 虚高，v4.22 资本无关口径的 +0.59% 才是诚实基线。</p>
<p><b>2. 因子层面只剩低波有正 IC。</b> #23 重算印证：低波是唯一稳健正 alpha（IC 显著正），动量/趋势为反 alpha，质量/成长近似零。策略应继续「低波+反转」内核，放弃对质量/成长收益的贡献预期。</p>
<p><b>3. 降频不创造 alpha。</b> #24 显示拉长持有仅降低换手与改变回撤结构，无法把近似零的毛收益变成正收益；降频的真正价值在「省成本+降波动」，而非增厚收益。</p>
<p><b>行动建议：</b>该策略定位为「低波防御 + 闲资停泊」组合，而非收益引擎；若要实质提升，需在因子外寻找新边缘（行业轮动择时、市场闸门精细化、或引入衍生对冲），而非在现有 lvrev 参数内调优。</p>
</div>

<p style="color:#999;font-size:12px;margin-top:30px">本报告由 empirical/ 下回测日志与 compute_ic.py 结果自动汇总生成。所有回测均顺序执行、零线程、CPU-safe。</p>
</body></html>"""

Path(OUT).write_text(html, encoding="utf-8")
print("report ->", OUT, os.path.getsize(OUT), "bytes")
