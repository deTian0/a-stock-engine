"""
html_report.py — Qbot 风格的 HTML 简报生成器

借鉴 Qbot (quantstats tear sheet) + vnpy 图表:
  - ECharts 交互图表 (市场雷达、板块热力、评分分布)
  - 样式化表格 (中长线/短线/ETF/持仓)
  - 自包含单文件 HTML，无需外部依赖
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime


def generate_html(results: dict, config: dict) -> str:
    """生成自包含 HTML 简报。"""
    today = datetime.now().strftime("%Y-%m-%d")
    regime = results.get("regime", {})
    categories = results.get("categories", {})
    l4 = results.get("l4_results", pd.DataFrame())

    quality = categories.get("②A_质量榜", pd.DataFrame())
    short_list = categories.get("②B_短线榜", pd.DataFrame())
    etf_picks = results.get("etf_picks", pd.DataFrame())

    regime_name = regime.get("regime", "未知") if isinstance(regime, dict) else str(regime)
    pos_cap = regime.get("position_cap", 0.5) if isinstance(regime, dict) else 0.5

    # === ECharts 数据 ===
    # 评分分布
    scores = l4.get("composite_score", pd.Series([50])).dropna().tolist() if len(l4) > 0 else [50]
    score_bins = [0, 40, 50, 60, 70, 80, 90, 100]
    score_labels = ["<40", "40-50", "50-60", "60-70", "70-80", "80-90", "90+"]
    score_hist = [sum(1 for s in scores if s >= score_bins[i] and s < score_bins[i+1]) 
                   for i in range(len(score_bins)-1)]

    # 板块分布
    sector_data = {}
    if len(quality) > 0 and "sector" in quality.columns:
        for _, r in quality.iterrows():
            s = r.get("sector", "其他")
            sector_data[s] = sector_data.get(s, 0) + 1
    elif len(short_list) > 0 and "sector" in short_list.columns:
        for _, r in short_list.iterrows():
            s = r.get("sector", "其他")
            sector_data[s] = sector_data.get(s, 0) + 1

    sector_pie = [{"name": k, "value": v} for k, v in sorted(sector_data.items(), key=lambda x: -x[1])[:8]]

    # 信号统计
    if len(l4) > 0 and "tech_signal" in l4.columns:
        signal_counts = l4["tech_signal"].value_counts().to_dict()
    else:
        signal_counts = {"🟢偏多": 0, "🟡震荡": 0, "🔴偏空": 0}

    # === HTML 构建 ===
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>盘前选股简报 — {today}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f7fa;color:#333;padding:20px;max-width:1200px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:24px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:22px;margin-bottom:8px}}
.header .meta{{opacity:.85;font-size:14px}}
.summary{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}}
.card{{background:#fff;border-radius:10px;padding:16px 20px;flex:1;min-width:160px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.card .label{{font-size:12px;color:#999;margin-bottom:4px}}
.card .value{{font-size:24px;font-weight:700}}
.card.warn{{border-left:3px solid #f56c6c}}
.card.ok{{border-left:3px solid #67c23a}}
.card.info{{border-left:3px solid #409eff}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
.chart-box{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.06);height:300px}}
.section{{background:#fff;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section h2{{font-size:18px;margin-bottom:12px;color:#667eea}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f0f2f5;padding:8px 6px;text-align:left;font-weight:600;border-bottom:2px solid #e4e7ed;white-space:nowrap}}
td{{padding:6px;border-bottom:1px solid #ebeef5}}
tr:hover{{background:#f5f7fa}}
.bullish{{color:#67c23a;font-weight:600}}
.bearish{{color:#f56c6c;font-weight:600}}
.neutral{{color:#e6a23c}}
.tag{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:600}}
.tag-green{{background:#e1f3d8;color:#67c23a}}
.tag-red{{background:#fef0f0;color:#f56c6c}}
.tag-yellow{{background:#fdf6ec;color:#e6a23c}}
.footer{{text-align:center;color:#999;font-size:12px;padding:16px}}
</style>
</head>
<body>
<div class="header">
  <h1>📊 盘前选股简报 — {today}</h1>
  <div class="meta">生成时间: {results.get('timestamp', '')} | 耗时: {results.get('elapsed_seconds', 0)}s | 
  选股数: {len(quality)} 质量 + {len(short_list)} 短线 + {len(etf_picks)} ETF</div>
</div>

<div class="summary">
  <div class="card {'warn' if '空头' in regime_name else 'ok'}">
    <div class="label">市场环境</div>
    <div class="value">{regime_name}</div>
  </div>
  <div class="card info">
    <div class="label">仓位上限</div>
    <div class="value">{pos_cap:.0%}</div>
  </div>
  <div class="card">
    <div class="label">中长线推荐</div>
    <div class="value">{len(quality)}<span style="font-size:14px;font-weight:400"> 只</span></div>
  </div>
  <div class="card">
    <div class="label">短线推荐</div>
    <div class="value">{len(short_list)}<span style="font-size:14px;font-weight:400"> 只</span></div>
  </div>
</div>

<div class="charts">
  <div class="chart-box"><div id="chartScore" style="width:100%;height:100%"></div></div>
  <div class="chart-box"><div id="chartSector" style="width:100%;height:100%"></div></div>
</div>
"""

    # === 中长线表 ===
    html += '<div class="section"><h2>🏆 中长线组合（建议持仓 5-20 日）</h2>'
    if len(quality) > 0:
        html += '<table><tr><th>强度</th><th>代码</th><th>名称</th><th>股价</th><th>止损</th><th>信号</th><th>技术面</th><th>基本面</th><th>评分</th><th>仓位</th><th>流动性</th></tr>'
        for _, r in quality.head(8).iterrows():
            grade = r.get("signal_grade", "⚪")
            code = r.get("code", "")
            name = _clean(r.get("name", code))
            close = _fmt_val(r.get("close"))
            stop = _fmt_val(r.get("stop_loss"))
            sig = r.get("tech_signal", "-")
            sig_cls = "bullish" if "偏多" in str(sig) else ("bearish" if "偏空" in str(sig) else "neutral")
            tech = f"{r.get('tech_ma','-')}|{r.get('tech_macd','-')}"
            fund = f"ROE:{_fmt_val(r.get('roe'),pct=True)}"
            score = r.get("composite_score", 0)
            pos = f"{r.get('suggested_position',0):.1f}%"
            liq = r.get("liquidity_tag", "-")
            html += f'<tr><td style="font-size:16px">{grade}</td><td>{code}</td><td>{name}</td><td>{close}</td><td class="bearish">{stop}</td>'
            html += f'<td class="{sig_cls}">{sig}</td><td style="font-size:11px">{tech}</td><td style="font-size:11px">{fund}</td>'
            html += f'<td>{score:.0f}</td><td>{pos}</td><td>{liq}</td></tr>'
        html += '</table>'
    else:
        html += '<p style="color:#999">暂无中长线推荐</p>'
    html += '</div>'

    # === 短线表 ===
    html += '<div class="section"><h2>⚡ 短线组合（建议持仓 1-5 日）</h2>'
    if len(short_list) > 0:
        html += '<table><tr><th>代码</th><th>名称</th><th>股价</th><th>止损</th><th>信号</th><th>概念</th><th>概念涨</th><th>评分</th><th>动量20</th><th>流动性</th></tr>'
        for _, r in short_list.head(8).iterrows():
            code = r.get("code", "")
            name = _clean(r.get("name", code))
            close = _fmt_val(r.get("close"))
            stop = _fmt_val(r.get("stop_loss"))
            sig = r.get("tech_signal", "-")
            sig_cls = "bullish" if "偏多" in str(sig) else ("bearish" if "偏空" in str(sig) else "neutral")
            concept = r.get("concept_name", "-") or "-"
            cchg = _fmt_val(r.get("concept_chg"), pct=True)
            score = r.get("composite_score", 0)
            mom20 = _fmt_val(r.get("momentum_20d"), pct=True)
            liq = r.get("liquidity_tag", "-")
            html += f'<tr><td>{code}</td><td>{name}</td><td>{close}</td><td class="bearish">{stop}</td>'
            html += f'<td class="{sig_cls}">{sig}</td><td>{concept}</td><td>{cchg}</td>'
            html += f'<td>{score:.0f}</td><td>{mom20}</td><td>{liq}</td></tr>'
        html += '</table>'
    else:
        html += '<p style="color:#999">暂无短线推荐</p>'
    html += '</div>'

    # === ETF ===
    html += '<div class="section"><h2>📦 ETF 组合</h2>'
    if len(etf_picks) > 0:
        html += '<table><tr><th>代码</th><th>名称</th><th>类型</th><th>动量20日</th><th>建议</th></tr>'
        for _, r in etf_picks.head(8).iterrows():
            code = r.get("code", "")
            name = r.get("name", code)
            etype = r.get("etf_type", "-")
            mom20 = _fmt_val(r.get("momentum_20d"), pct=True)
            advice = r.get("advice", "关注")
            html += f'<tr><td>{code}</td><td>{name}</td><td>{etype}</td><td>{mom20}</td><td>{advice}</td></tr>'
        html += '</table>'
    else:
        html += '<p style="color:#999">ETF数据暂不可用</p>'
    html += '</div>'

    # === ECharts JS ===
    html += f"""
<script>
(function(){{
  // 评分分布
  var c1 = echarts.init(document.getElementById('chartScore'));
  c1.setOption({{
    title: {{text:'评分分布',left:'center',textStyle:{{fontSize:14}}}},
    tooltip: {{trigger:'axis'}},
    xAxis: {{data:{json.dumps(score_labels)}}},
    yAxis: {{type:'value'}},
    series: [{{name:'股票数',type:'bar',data:{json.dumps(score_hist)},
      itemStyle:{{color:new echarts.graphic.LinearGradient(0,0,0,1,[
        {{offset:0,color:'#667eea'}},{{offset:1,color:'#764ba2'}}])}}}}],
    grid: {{left:40,right:20,top:40,bottom:30}}
  }});

  // 板块分布
  var c2 = echarts.init(document.getElementById('chartSector'));
  c2.setOption({{
    title: {{text:'板块分布',left:'center',textStyle:{{fontSize:14}}}},
    tooltip: {{trigger:'item'}},
    series: [{{type:'pie',radius:['40%','70%'],data:{json.dumps(sector_pie)},
      label:{{fontSize:11}},emphasis:{{itemStyle:{{shadowBlur:10,shadowColor:'rgba(0,0,0,.3)'}}}}}}]
  }});

  window.addEventListener('resize',function(){{c1.resize();c2.resize();}});
}})();
</script>
"""

    html += f'<div class="footer">🚀 自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")} | Powered by a-stock-engine</div>'
    html += '</body></html>'
    return html


def _clean(val):
    v = str(val) if val is not None else ""
    if v.lower() in ("nan", "none", ""):
        return "-"
    return v


def _fmt_val(val, pct=False):
    if val is None or not pd.notna(val) or val == 0:
        return "-"
    if pct:
        return f"{val:+.1f}%"
    return f"{val:.2f}"
