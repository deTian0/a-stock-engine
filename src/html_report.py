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
from risk_module import allocate_basket


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

    # 篮子分配: 中长线仓位改为占总资金比例 (修复单票按 position_cap 满算)
    brief_cfg = config.get("brief", {}) if isinstance(config, dict) else {}
    sleeve_weights = brief_cfg.get("sleeve_weights", {"quality": 0.50, "short_term": 0.30, "etf": 0.20})
    alloc_method = brief_cfg.get("method", "score_weighted")
    max_single = brief_cfg.get("max_single_position", 0.08)
    q_view = quality.head(8) if len(quality) > 0 else quality
    q_scores = q_view["composite_score"].tolist() if "composite_score" in q_view.columns else []
    q_budget = pos_cap * sleeve_weights.get("quality", 0.50)
    q_alloc = allocate_basket(q_scores, q_budget, method=alloc_method, max_single=max_single)
    q_pos_map = {c: a for c, a in zip(q_view["code"].tolist(), q_alloc)} if "code" in q_view.columns else {}

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
.codes-block{{margin:10px 0 14px;padding:10px 12px;background:#f7f9fc;border:1px solid #e3e8f0;border-radius:8px}}
.codes-label{{font-size:13px;color:#1f2d3d;margin-bottom:6px}}
.codes-text{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;line-height:1.6;white-space:pre-wrap;margin:0 0 8px;color:#0a4d8c}}
.copy-btn{{background:#1677ff;color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:13px;cursor:pointer}}
.copy-btn:hover{{background:#0958d9}}
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
            pos = f"{q_pos_map.get(r.get('code', ''), 0) * 100:.1f}%"
            liq = r.get("liquidity_tag", "-")
            html += f'<tr><td style="font-size:16px">{grade}</td><td>{code}</td><td>{name}</td><td>{close}</td><td class="bearish">{stop}</td>'
            html += f'<td class="{sig_cls}">{sig}</td><td style="font-size:11px">{tech}</td><td style="font-size:11px">{fund}</td>'
            html += f'<td>{score:.0f}</td><td>{pos}</td><td>{liq}</td></tr>'
        html += '</table>'
    else:
        html += '<p style="color:#999">暂无中长线推荐</p>'
    html += '</div>'
    html += _html_codes_block(quality, "中长线")

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
    html += _html_codes_block(short_list, "短线")

    # === ETF ===
    html += '<div class="section"><h2>📦 ETF 组合（A股股票型ETF为 T+1，仅货币/债券/黄金/跨境ETF为 T+0）</h2>'
    if len(etf_picks) > 0:
        html += '<table><tr><th>代码</th><th>名称</th><th>类型</th><th>结算</th><th>动量20日</th><th>建议</th></tr>'
        for _, r in etf_picks.head(8).iterrows():
            code = r.get("code", "")
            name = r.get("name", code)
            etype = r.get("etf_type", "-")
            settle = _etf_settlement(code)
            mom20 = _fmt_val(r.get("momentum_20d"), pct=True)
            advice = r.get("advice", "关注")
            html += f'<tr><td>{code}</td><td>{name}</td><td>{etype}</td><td>{settle}</td><td>{mom20}</td><td>{advice}</td></tr>'
        html += '</table>'
    else:
        html += '<p style="color:#999">ETF数据暂不可用</p>'
    html += '</div>'
    html += _html_codes_block(etf_picks, "ETF")

    # === 观察名单（来自 categories ③C_观察名单）===
    watchlist = categories.get("③C_观察名单")
    if isinstance(watchlist, pd.DataFrame) and len(watchlist) > 0:
        html += '<div class="section"><h2>👀 观察名单</h2>'
        html += '<table><tr><th>代码</th><th>名称</th><th>板块</th><th>概念</th><th>评分</th><th>关注理由</th></tr>'
        for _, r in watchlist.head(20).iterrows():
            code = r.get("code", "")
            name = _clean(r.get("name", code))
            sector = r.get("sector", "-")
            concept = r.get("concept_name", "-") or "-"
            score = r.get("composite_score", 0)
            reason = r.get("reason", "综合因子")
            html += f'<tr><td>{code}</td><td>{name}</td><td>{sector}</td><td>{concept}</td><td>{score:.0f}</td><td>{reason}</td></tr>'
        html += '</table>'
        html += '</div>'
        html += _html_codes_block(watchlist, "观察")

    # === 持仓追踪（来自 config.account.holdings + results.holding_prices）===
    holdings = config.get("account", {}).get("holdings", {}) if isinstance(config, dict) else {}
    if holdings:
        holding_prices = results.get("holding_prices", {}) or {}
        l4 = results.get("l4_results", pd.DataFrame())
        cur_price_map = {}
        if isinstance(l4, pd.DataFrame) and len(l4) > 0 and "code" in l4.columns and "close" in l4.columns:
            for _, r in l4.iterrows():
                cur_price_map[str(r["code"]).zfill(6)] = r["close"]
        total_assets = config.get("account", {}).get("total_assets", 0)
        cash = config.get("account", {}).get("available_cash", 0)
        pos_cap = regime.get("position_cap", 0.5) if isinstance(regime, dict) else 0.5

        html += '<div class="section"><h2>📈 持仓追踪与建议</h2>'
        html += '<table><tr><th>代码</th><th>名称</th><th>当日股价</th><th>成本</th><th>盈亏</th><th>持仓数</th><th>市值</th><th>建议</th></tr>'
        total_mv = 0
        hold_codes = []
        for code, info in holdings.items():
            code_str = str(code).zfill(6)
            hold_codes.append(code_str)
            shares = info.get("shares", 0)
            cost = info.get("cost_price", 0)
            name = info.get("name", code_str)
            cur_price = holding_prices.get(code_str)
            if cur_price is None:
                cur_price = cur_price_map.get(code_str, cost)
            pnl = (cur_price - cost) * shares
            pnl_pct = ((cur_price / cost) - 1) * 100 if cost > 0 else 0
            mv = cur_price * shares
            total_mv += mv
            if pnl_pct > 5:
                advice = "✅ 持有"
            elif pnl_pct > -3:
                advice = "🟢 持平"
            else:
                advice = "⚠️ 关注"
            html += f'<tr><td>{code_str}</td><td>{name}</td><td>{cur_price:.3f}</td><td>{cost:.3f}</td>' \
                    f'<td>{pnl:+.1f}({pnl_pct:+.1f}%)</td><td>{shares}</td><td>{mv:.0f}</td><td>{advice}</td></tr>'
        html += '</table>'
        position_pct = (total_mv / total_assets * 100) if total_assets > 0 else 0
        target_pos = pos_cap * 100
        html += f'<p><b>当前状态</b>: 总持仓 {total_mv:.0f} 元 | 仓位 {position_pct:.1f}% | 可用资金 {cash:.0f} 元</p>'
        if position_pct < target_pos - 5:
            advice_line = f"仓位低于市场允许上限({target_pos:.0f}%)可加仓"
        elif position_pct > target_pos + 5:
            advice_line = f"仓位高于市场允许上限({target_pos:.0f}%)应减仓"
        else:
            advice_line = f"仓位处于市场允许上限({target_pos:.0f}%)附近"
        html += f'<p><b>操作建议</b>: {advice_line}</p>'
        html += '</div>'
        html += _html_codes_block(hold_codes, "持仓")

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
    html += '<script>function copyCodes(btn){var c=btn.getAttribute("data-codes")||"";if(navigator.clipboard){navigator.clipboard.writeText(c).then(function(){var t=btn.textContent;btn.textContent="已复制";setTimeout(function(){btn.textContent=t;},1500);});}else{var ta=document.createElement("textarea");ta.value=c;document.body.appendChild(ta);ta.select();document.execCommand("copy");document.body.removeChild(ta);btn.textContent="已复制";setTimeout(function(){btn.textContent="复制全部";},1500);}}</script>'
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


def _etf_settlement(code: str) -> str:
    """A股 ETF 结算方式：货币(511)/债券(511)/黄金(518)/跨境(513) 等为 T+0，
    其余宽基/行业/主题等境内股票型ETF 均为 T+1。与 daily_brief 逻辑保持一致。"""
    c = str(code).zfill(6)
    if c.startswith(("511", "518", "513")):
        return "T+0"
    return "T+1"


def _html_codes_block(codes, label=""):
    """生成可一键复制到同花顺自选的代码块（与 MD 简报同逻辑：每行最多5个）。

    入参 codes 可为本节 DataFrame（自动安全提取 'code' 列）或代码列表。
    """
    if isinstance(codes, pd.DataFrame):
        if len(codes) == 0 or "code" not in codes.columns:
            return ""
        codes = codes["code"].tolist()
    norm = [str(c).zfill(6) for c in (codes or []) if c]
    if not norm:
        return ""
    joined = " ".join(norm)
    lines = [" ".join(norm[i:i + 5]) for i in range(0, len(norm), 5)]
    disp = "\n".join(lines)
    lbl = f" · {label}" if label else ""
    return (
        '<div class="codes-block">'
        f'<div class="codes-label">📋 <b>同花顺自选（一键复制）</b>{lbl}</div>'
        f'<pre class="codes-text">{disp}</pre>'
        f'<button class="copy-btn" data-codes="{joined}" onclick="copyCodes(this)">复制全部</button>'
        '</div>'
    )
