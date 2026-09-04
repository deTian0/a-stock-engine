#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""个股深度调研报告生成器 (stock_research)

用法:
    python -m src.stock_research 000533
    python -m src.stock_research 000533 600751

输出 (history/research/YYYY-MM-DD/):
    {code}_调研.md      Markdown 报告（每节含同花顺自选复制块，表格外、每行≤5码）
    {code}_调研.html    HTML 报告（含 ECharts K线+MA / MACD / 量能 三图 + 一键复制按钮）

数据源:
    实时行情/估值 : 腾讯 qt.gtimg.cn 直连（最全，字段已交叉验证）
    技术面 K线     : westock_cli.get_kline (westock → tushare → akshare 兜底)
    财务 ROE/增长 : 本地 fundamentals 表 → tushare fina_indicator 兜底
    板块/概念      : westock profile (business / sector)
    大盘环境       : 三大指数 get_index_kline 算 MA60 状态
"""
import sys, os, io, json, argparse, datetime, logging

sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import requests
from westock_cli import WestockCLI
from database import StockDB

logger = logging.getLogger("stock_research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CLI = WestockCLI()
DB = StockDB()
TODAY = datetime.date.today()
OUT_DIR = os.path.join("history", "research", TODAY.strftime("%Y-%m-%d"))


# ============================================================
#  工具
# ============================================================
def _safe(x, default="-"):
    try:
        if x is None:
            return default
        if isinstance(x, float) and pd.isna(x):
            return default
        return x
    except Exception:
        return default


def _f(x, nd=2, default="-"):
    v = _safe(x, None)
    if v is None or v == "":
        return default
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return default


def _amt(s):
    try:
        v = float(s)
        if v >= 1e8:
            return f"{v / 1e8:.2f}亿"
        if v >= 1e4:
            return f"{v / 1e4:.2f}万"
        return f"{v:.0f}元"
    except Exception:
        return "-"


def _codes_block_md(code: str) -> str:
    """MD 同花顺复制块：表格外一行，每行≤5码。"""
    return f"> 📋 **同花顺自选(复制)**: {code.zfill(6)}"


# ============================================================
#  1. 腾讯实时行情（字段已交叉验证）
# ============================================================
def tencent_quote(code: str) -> dict:
    prefix = "sh" if code.startswith("6") else ("sz" if code.startswith(("0", "3")) else "bj")
    url = f"https://qt.gtimg.cn/q={prefix}{code}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=12, proxies={"http": None, "https": None})
        body = r.text.split('="', 1)[1].rstrip(';\n')
        f = body.split("~")

        def g(i):
            return f[i] if i < len(f) else ""

        amt = g(35).split("/")[-1] if "/" in g(35) else ""
        return {
            "name": g(1), "price": _f(g(3)), "prev_close": _f(g(4)), "open": _f(g(5)),
            "change": _f(g(31)), "pct": _f(g(32)), "high": _f(g(33)), "low": _f(g(34)),
            "amount": amt, "turnover": _f(g(38)), "pe": _f(g(39)), "amplitude": _f(g(43)),
            "circ_mv": _f(g(44)), "total_mv": _f(g(45)), "pb": _f(g(46)),
            "limit_up": _f(g(47)), "limit_down": _f(g(48)), "vol_ratio": _f(g(49)),
            "time": g(30),
        }
    except Exception as e:
        logger.warning(f"腾讯行情失败 {code}: {e}")
        return {}


# ============================================================
#  2. 技术面（K线重算）
# ============================================================
def compute_technical(df, days=120):
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    n = len(close)

    def ma(w):
        return close.rolling(w).mean() if n >= w else pd.Series([np.nan] * n)

    ma5, ma10, ma20, ma60 = ma(5), ma(10), ma(20), ma(60)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    price = close.iloc[-1]

    def pct_above(m):
        return (price / m - 1) * 100 if pd.notna(m) and m > 0 else float("nan")

    def mom(w):
        return (price / close.iloc[-w] - 1) * 100 if n > w else float("nan")

    sl = df["date"].tolist()[-days:]
    o = pd.to_numeric(df["open"], errors="coerce").tolist()[-days:]
    c = close.tolist()[-days:]
    h = pd.to_numeric(df["high"], errors="coerce").tolist()[-days:]
    l = pd.to_numeric(df["low"], errors="coerce").tolist()[-days:]
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0).tolist()[-days:]
    ohlc = [[_safe(x, 0) for x in [o[i], c[i], l[i], h[i]]] for i in range(len(sl))]

    def trim(s):
        return [None if pd.isna(x) else round(float(x), 3) for x in s.tolist()[-days:]]

    return {
        "price": price,
        "ma5": ma5.iloc[-1], "ma10": ma10.iloc[-1], "ma20": ma20.iloc[-1], "ma60": ma60.iloc[-1],
        "above_ma20": pct_above(ma20.iloc[-1]), "above_ma60": pct_above(ma60.iloc[-1]),
        "dif": dif.iloc[-1], "dea": dea.iloc[-1], "macd": macd.iloc[-1],
        "macd_cross": "金叉(红柱)" if macd.iloc[-1] > 0 else "死叉(绿柱)",
        "mom5": mom(5), "mom20": mom(20), "mom60": mom(60),
        "recent": [round(float(x), 2) for x in c[-10:]],
        "series": {
            "date": sl, "ohlc": ohlc, "vol": [round(float(x), 0) for x in v],
            "ma20": trim(ma20), "ma60": trim(ma60),
            "dif": trim(dif), "dea": trim(dea), "macd": trim(macd),
        },
    }


# ============================================================
#  3. 财务（本地 → tushare 兜底）
# ============================================================
def fetch_fundamentals(code: str) -> dict:
    df = DB.get_fundamentals_table([code])
    if df is not None and len(df):
        r = df.iloc[0]
        return {k: r.get(k) for k in
                ["roe", "roa", "gross_margin", "debt_ratio",
                 "revenue_growth", "profit_growth", "pe", "pb", "total_mv", "circ_mv"]}
    try:
        if CLI._ts is not None:
            fin = CLI._ts.get_fundamentals([code])
            if fin is not None and len(fin):
                r = fin.iloc[0]
                return {"roe": r.get("roe"), "roa": r.get("roa"),
                        "gross_margin": r.get("gross_margin"), "debt_ratio": r.get("debt_ratio"),
                        "revenue_growth": r.get("revenue_growth"), "profit_growth": r.get("profit_growth"),
                        "pe": r.get("pe"), "pb": r.get("pb"), "total_mv": r.get("market_cap")}
    except Exception as e:
        logger.warning(f"tushare 财务失败 {code}: {e}")
    return {}


# ============================================================
#  4. 板块 / 概念
# ============================================================
def fetch_sector(code: str):
    try:
        prof = CLI.get_fundamentals([code])
        if prof is not None and len(prof):
            prof = prof.loc[:, ~prof.columns.duplicated()]  # westock profile 偶发重复列
            sec = prof["sector"].iloc[0] if "sector" in prof.columns else None
            biz = prof["business"].iloc[0] if "business" in prof.columns else None
            return _safe(sec, "-"), _safe(biz, "-")
    except Exception:
        pass
    return "-", "-"


# ============================================================
#  5. 大盘环境（三大指数 MA60 状态）
# ============================================================
def market_env():
    res = []
    for idx, name in [("000001", "上证指数"), ("399001", "深成指"), ("399006", "创业板指")]:
        try:
            df = CLI.get_index_kline(idx, days=65)
            close = pd.to_numeric(df["close"], errors="coerce")
            ma60 = close.rolling(60).mean().iloc[-1]
            res.append((name, close.iloc[-1], ma60, bool(close.iloc[-1] > ma60)))
        except Exception:
            res.append((name, float("nan"), float("nan"), False))
    return res


def _market_label(mkt):
    up = sum(1 for *_, x in mkt if x)
    return ("多头(3/3站上MA60)" if up == 3 else
            "结构性偏多(1/3站上MA60)" if up == 1 else
            "震荡(2/3站上MA60)" if up == 2 else
            "空头(0/3站上MA60)")


# ============================================================
#  6. 评级 / 风险 / 建议
# ============================================================
def _assess(code, q, tech, fin, mkt):
    above_ma60 = tech and pd.notna(tech["above_ma60"]) and tech["above_ma60"] > 0
    above_ma20 = tech and pd.notna(tech["above_ma20"]) and tech["above_ma20"] > 0
    macd_pos = tech and tech["macd"] > 0
    up = sum(1 for *_, x in mkt if x)
    mkt_txt = _market_label(mkt)

    if above_ma60 and above_ma20:
        strength = "⚡ 中长线强势（站上 MA20 且站上 MA60）"
    elif above_ma20:
        strength = "🔥 短线强势（站上 MA20，MA60 下方）"
    elif above_ma60:
        strength = "震荡偏强（站上 MA60，MA20 下方）"
    else:
        strength = "🐻 弱势（MA20/MA60 双下方）"

    if above_ma60 and above_ma20 and macd_pos:
        conclusion = "强势独立股，趋势与动能俱佳；但高位需防回踩，建议回踩 MA20 企稳小仓参与，不追高"
    elif above_ma20:
        conclusion = f"短线偏强但中期未确认，MACD{'红柱' if macd_pos else '绿柱'}；建议轻仓试探、严格止损"
    else:
        conclusion = "当前弱势（跌破主要均线），不建议抄底；等待重新站上 MA20/MA60 再考虑"

    risks = []
    if tech and pd.notna(tech["above_ma60"]) and tech["above_ma60"] > 10:
        risks.append(f"已显著高出 MA60（{tech['above_ma60']:.1f}%），追高风险大")
    if tech and tech["macd"] <= 0:
        risks.append("MACD 绿柱/死叉，短期动能减弱")
    if q.get("turnover") not in (None, "-", "") and float(q["turnover"]) > 10:
        risks.append(f"换手率 {q['turnover']}% 偏高，筹码交换剧烈")
    if fin.get("roe") is not None and pd.notna(fin.get("roe")) and fin["roe"] < 5:
        risks.append(f"ROE 仅 {fin['roe']:.1f}%，质量成色不足")
    if up == 0:
        risks.append("大盘空头（0/3 指数站上 MA60），系统性风险未解除")
    elif up < 3:
        risks.append(f"大盘{mkt_txt}，结构性机会非全面牛市")

    advice = []
    if tech:
        ma20, ma60, price = tech["ma20"], tech["ma60"], tech["price"]
        recent_low = min(tech["recent"]) if tech["recent"] else price
        advice.append(f"支撑位：MA20≈{ma20:.2f}、MA60≈{ma60:.2f}、前低≈{recent_low:.2f}")
        advice.append("压力位：近期高点（现价上方）")
        advice.append(f"止损线：跌破 MA60({ma60:.2f}) 或前低({recent_low:.2f}) 减仓")
        advice.append(f"买点：回踩 MA20({ma20:.2f}) 企稳且 MACD 翻红，再小仓")
    advice.append(f"仓位纪律：单票≤10%、总仓≤20%（当前市场「{mkt_txt}」），轻仓防御为主")

    return conclusion, strength, risks, advice


# ============================================================
#  7. MD 报告
# ============================================================
def build_md(code, q, tech, fin, sector, business, mkt):
    name = q.get("name") or code
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mkt_txt = _market_label(mkt)
    conclusion, strength, risks, advice = _assess(code, q, tech, fin, mkt)

    L = []
    L.append(f"# 个股调研：{code} {name}\n")
    L.append(f"> 生成时间：{now} ｜ 数据：腾讯实时 + westock K线 + tushare 财务 + 三大指数")
    L.append(f"> **结论：{conclusion}**\n")

    # 一、实时快照
    L.append("## 一、实时快照")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| 现价 | {q.get('price', '-')} |")
    L.append(f"| 涨跌幅 | {q.get('pct', '-')}% |")
    L.append(f"| 今开 / 昨收 | {q.get('open', '-')} / {q.get('prev_close', '-')} |")
    L.append(f"| 最高 / 最低 | {q.get('high', '-')} / {q.get('low', '-')} |")
    L.append(f"| 换手率 | {q.get('turnover', '-')}% |")
    L.append(f"| 量比 | {q.get('vol_ratio', '-')} |")
    L.append(f"| 成交额 | {_amt(q.get('amount'))} |")
    L.append(f"| 振幅 | {q.get('amplitude', '-')}% |")
    L.append(f"| PE(TTM) | {q.get('pe', '-')} |")
    L.append(f"| PB | {q.get('pb', '-')} |")
    L.append(f"| 总市值 | {q.get('total_mv', '-')} 亿 |")
    L.append(f"| 流通市值 | {q.get('circ_mv', '-')} 亿 |")
    L.append(f"| 涨停 / 跌停 | {q.get('limit_up', '-')} / {q.get('limit_down', '-')} |")
    L.append(f"| 盘口时间 | {q.get('time', '-')} |")

    # 二、技术面
    L.append("\n## 二、技术面")
    L.append(f"**强度评级：{strength}**\n")
    L.append("| 指标 | 值 | 解读 |")
    L.append("|------|-----|------|")
    if tech:
        L.append(f"| 现价 | {_f(tech['price'])} | - |")
        L.append(f"| MA20 | {_f(tech['ma20'])} | 相对 {_f(tech['above_ma20'])}% |")
        L.append(f"| MA60 | {_f(tech['ma60'])} | 相对 {_f(tech['above_ma60'])}% |")
        L.append(f"| MACD | DIF {_f(tech['dif'])} / DEA {_f(tech['dea'])} / 柱 {_f(tech['macd'])} | {tech['macd_cross']} |")
        L.append(f"| 5日动量 | {_f(tech['mom5'])}% | - |")
        L.append(f"| 20日动量 | {_f(tech['mom20'])}% | - |")
        L.append(f"| 60日动量 | {_f(tech['mom60'])}% | - |")
        L.append(f"| 近10日收盘 | {' → '.join(_f(x) for x in tech['recent'])} | - |")
    else:
        L.append("| - | 数据缺失 | 网络/K线获取失败 |")
    L.append("")
    L.append(_codes_block_md(code))

    # 三、基本面
    L.append("\n## 三、基本面")
    L.append("### 估值（实时）")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| PE(TTM) | {q.get('pe', '-')} |")
    L.append(f"| PB | {q.get('pb', '-')} |")
    L.append(f"| 总市值 | {q.get('total_mv', '-')} 亿 |")
    L.append(f"| 流通市值 | {q.get('circ_mv', '-')} 亿 |")
    L.append("\n### 财务（最新报告期，本地缺失时由 tushare 补）")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| ROE | {_f(fin.get('roe'))}% |")
    L.append(f"| ROA | {_f(fin.get('roa'))}% |")
    L.append(f"| 毛利率 | {_f(fin.get('gross_margin'))}% |")
    L.append(f"| 负债率 | {_f(fin.get('debt_ratio'))}% |")
    L.append(f"| 营收增长 | {_f(fin.get('revenue_growth'))}% |")
    L.append(f"| 利润增长 | {_f(fin.get('profit_growth'))}% |")
    L.append("")
    L.append(_codes_block_md(code))

    # 四、板块与概念
    L.append("\n## 四、板块与概念")
    L.append(f"- **所属行业**：{sector}")
    L.append(f"- **主营业务**：{business}")
    L.append("")
    L.append(_codes_block_md(code))

    # 五、大盘环境对比
    L.append("\n## 五、大盘环境对比")
    L.append(f"**当前市场：{mkt_txt}**\n")
    L.append("| 指数 | 收盘 | MA60 | 站上MA60 |")
    L.append("|------|------|------|---------|")
    for nm, close, ma60, ok in mkt:
        L.append(f"| {nm} | {_f(close)} | {_f(ma60)} | {'✔ 站上' if ok else '✘ 下方'} |")
    L.append("")
    L.append(_codes_block_md(code))

    # 六、风险提示
    L.append("\n## 六、风险提示")
    if risks:
        for r in risks:
            L.append(f"- ⚠️ {r}")
    else:
        L.append("- 暂无显著风险点（以实时数据为准）")

    # 七、操作建议
    L.append("\n## 七、操作建议")
    for a in advice:
        L.append(f"- {a}")
    L.append("")
    L.append(_codes_block_md(code))

    # 免责
    L.append("\n---\n")
    L.append("> ⚠️ **免责声明**：本报告由程序基于公开行情数据自动生成，仅供研究参考，")
    L.append("> 不构成任何投资建议。股市有风险，买卖决策与盈亏由投资者自行承担。")
    L.append("> 实时数据请以券商终端为准。")
    return "\n".join(L)


# ============================================================
#  8. HTML 报告（ECharts + 一键复制）
# ============================================================
def _codes_block_html(code):
    return (f'<div class="codes-block">'
            f'<div class="codes-label">📋 <b>同花顺自选（一键复制）</b></div>'
            f'<pre class="codes-text">{code.zfill(6)}</pre>'
            f'<button class="copy-btn" data-codes="{code.zfill(6)}" onclick="copyCodes(this)">复制全部</button>'
            f'</div>')


def build_html(code, q, tech, fin, sector, business, mkt):
    name = q.get("name") or code
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mkt_txt = _market_label(mkt)
    conclusion, strength, risks, advice = _assess(code, q, tech, fin, mkt)

    # 快照卡片
    snap = (f'<div class="summary">'
            f'<div class="card"><div class="label">现价</div><div class="value">{q.get("price","-")}</div></div>'
            f'<div class="card {"ok" if q.get("pct","-") not in (None,"-") and float(q["pct"])>0 else "warn"}"><div class="label">涨跌幅</div><div class="value">{q.get("pct","-")}%</div></div>'
            f'<div class="card info"><div class="label">PE(TTM)</div><div class="value">{q.get("pe","-")}</div></div>'
            f'<div class="card info"><div class="label">总市值</div><div class="value">{q.get("total_mv","-")}亿</div></div>'
            f'<div class="card {"ok" if mkt_txt.startswith("多头") else "warn"}"><div class="label">大盘</div><div class="value" style="font-size:16px">{mkt_txt.split("(")[0]}</div></div>'
            f'</div>')

    # 技术面表
    if tech:
        tech_rows = [
            ["现价", _f(tech["price"]), "-"],
            ["MA20", _f(tech["ma20"]), f"相对 {_f(tech['above_ma20'])}%"],
            ["MA60", _f(tech["ma60"]), f"相对 {_f(tech['above_ma60'])}%"],
            ["MACD", f"DIF {_f(tech['dif'])} / DEA {_f(tech['dea'])} / 柱 {_f(tech['macd'])}", tech["macd_cross"]],
            ["20日动量", f"{_f(tech['mom20'])}%", "-"],
            ["60日动量", f"{_f(tech['mom60'])}%", "-"],
        ]
    else:
        tech_rows = [["-", "数据缺失", "网络/K线获取失败"]]
    tech_tbl = '<table><tr><th>指标</th><th>值</th><th>解读</th></tr>'
    for r in tech_rows:
        tech_tbl += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>"
    tech_tbl += "</table>"

    # 基本面表
    fund_tbl = ('<table><tr><th>类别</th><th>指标</th><th>值</th></tr>'
                f"<tr><td>估值</td><td>PE(TTM)</td><td>{q.get('pe','-')}</td></tr>"
                f"<tr><td>估值</td><td>PB</td><td>{q.get('pb','-')}</td></tr>"
                f"<tr><td>估值</td><td>总市值</td><td>{q.get('total_mv','-')}亿</td></tr>"
                f"<tr><td>财务</td><td>ROE</td><td>{_f(fin.get('roe'))}%</td></tr>"
                f"<tr><td>财务</td><td>毛利率</td><td>{_f(fin.get('gross_margin'))}%</td></tr>"
                f"<tr><td>财务</td><td>负债率</td><td>{_f(fin.get('debt_ratio'))}%</td></tr>"
                f"<tr><td>财务</td><td>营收增长</td><td>{_f(fin.get('revenue_growth'))}%</td></tr>"
                f"<tr><td>财务</td><td>利润增长</td><td>{_f(fin.get('profit_growth'))}%</td></tr>"
                "</table>")

    # 大盘表
    mkt_tbl = '<table><tr><th>指数</th><th>收盘</th><th>MA60</th><th>站上MA60</th></tr>'
    for nm, close, ma60, ok in mkt:
        mkt_tbl += f"<tr><td>{nm}</td><td>{_f(close)}</td><td>{_f(ma60)}</td><td>{'✔ 站上' if ok else '✘ 下方'}</td></tr>"
    mkt_tbl += "</table>"

    risk_html = "".join(f"<li>⚠️ {r}</li>" for r in risks) or "<li>暂无显著风险点</li>"
    advice_html = "".join(f"<li>{a}</li>" for a in advice)

    # ECharts 数据
    if tech:
        s = tech["series"]
        date_json = json.dumps(s["date"], ensure_ascii=False)
        ohlc_json = json.dumps(s["ohlc"])
        vol_json = json.dumps(s["vol"])
        ma20_json = json.dumps(s["ma20"])
        ma60_json = json.dumps(s["ma60"])
        dif_json = json.dumps(s["dif"])
        dea_json = json.dumps(s["dea"])
        macd_json = json.dumps(s["macd"])
        charts = """
        <div class="charts">
          <div class="chart-box"><div id="chartK" style="width:100%;height:100%"></div></div>
          <div class="chart-box"><div id="chartMacd" style="width:100%;height:100%"></div></div>
          <div class="chart-box" style="grid-column:1/3"><div id="chartVol" style="width:100%;height:100%"></div></div>
        </div>
        <script>
        var k=echarts.init(document.getElementById('chartK'));
        k.setOption({backgroundColor:'#fff',grid:{left:55,right:15,top:15,bottom:25},
          xAxis:{type:'category',data:__DATE__,axisLabel:{fontSize:9},
          yAxis:{scale:true},
          dataZoom:[{type:'inside'},{type:'slider',height:14,bottom:2}],
          legend:{data:['MA20','MA60'],top:0},
          series:[
            {type:'candlestick',name:'K',data:__OHLC__,
             itemStyle:{color:'#ef232a',color0:'#14b143',borderColor:'#ef232a',borderColor0:'#14b143'},
            {type:'line',name:'MA20',data:__MA20__,smooth:true,showSymbol:false,lineStyle:{width:1,color:'#f59e0b'},
            {type:'line',name:'MA60',data:__MA60__,smooth:true,showSymbol:false,lineStyle:{width:1,color:'#3b82f6'}
          ]});
        var m=echarts.init(document.getElementById('chartMacd'));
        m.setOption({backgroundColor:'#fff',grid:{left:55,right:15,top:15,bottom:25},
          xAxis:{type:'category',data:__DATE__,axisLabel:{fontSize:9},
          yAxis:{scale:true},
          legend:{data:['DIF','DEA'],top:0},
          series:[
            {type:'bar',name:'MACD',data:__MACD__,itemStyle:{color:function(p){return p.data>=0?'#ef232a':'#14b143';}},
            {type:'line',name:'DIF',data:__DIF__,showSymbol:false,lineStyle:{width:1,color:'#f59e0b'},
            {type:'line',name:'DEA',data:__DEA__,showSymbol:false,lineStyle:{width:1,color:'#3b82f6'}
          ]});
        var v=echarts.init(document.getElementById('chartVol'));
        v.setOption({backgroundColor:'#fff',grid:{left:55,right:15,top:15,bottom:25},
          xAxis:{type:'category',data:__DATE__,axisLabel:{fontSize:9},
          yAxis:{scale:true},
          series:[{type:'bar',name:'量',data:__VOL__,itemStyle:{color:'#94a3b8'}]});
        window.addEventListener('resize',function(){k.resize();m.resize();v.resize();});
        </script>
        """
        charts = (charts.replace("__DATE__", date_json).replace("__OHLC__", ohlc_json)
                  .replace("__MA20__", ma20_json).replace("__MA60__", ma60_json)
                  .replace("__DIF__", dif_json).replace("__DEA__", dea_json)
                  .replace("__MACD__", macd_json).replace("__VOL__", vol_json))
    else:
        charts = '<div class="section"><p>K线数据缺失，图表未生成。</p></div>'

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>个股调研 __CODE__ __NAME__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f7fa;color:#333;padding:20px;max-width:1200px;margin:0 auto}
.header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:24px;border-radius:12px;margin-bottom:20px}
.header h1{font-size:22px;margin-bottom:8px}
.header .meta{opacity:.85;font-size:14px}
.header .concl{margin-top:10px;font-size:15px;background:rgba(255,255,255,.15);padding:10px;border-radius:8px}
.summary{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}
.card{background:#fff;border-radius:10px;padding:16px 20px;flex:1;min-width:140px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.card .label{font-size:12px;color:#999;margin-bottom:4px}
.card .value{font-size:22px;font-weight:700}
.card.warn{border-left:3px solid #f56c6c}
.card.ok{border-left:3px solid #67c23a}
.card.info{border-left:3px solid #409eff}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.chart-box{background:#fff;border-radius:10px;padding:8px;box-shadow:0 2px 8px rgba(0,0,0,.06);height:300px}
.section{background:#fff;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.section h2{font-size:18px;margin-bottom:12px;color:#667eea}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f0f2f5;padding:8px 6px;text-align:left;font-weight:600;border-bottom:2px solid #e4e7ed;white-space:nowrap}
td{padding:6px;border-bottom:1px solid #ebeef5}
tr:hover{background:#f5f7fa}
.bullish{color:#67c23a;font-weight:600}
.bearish{color:#f56c6c;font-weight:600}
.codes-block{margin-top:14px;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;padding:12px}
.codes-label{font-size:13px;color:#475569;margin-bottom:6px}
.codes-text{font-family:Consolas,monospace;font-size:15px;letter-spacing:2px;color:#1e293b;margin:0 0 8px}
.copy-btn{background:#667eea;color:#fff;border:none;border-radius:6px;padding:7px 16px;font-size:13px;cursor:pointer}
.copy-btn:hover{background:#764ba2}
.risk li{margin:4px 0;color:#b45309}
.advice li{margin:4px 0;color:#1e40af}
</style>
</head>
<body>
<div class="header">
  <h1>个股调研：__CODE__ __NAME__</h1>
  <div class="meta">生成时间：__NOW__ ｜ 数据：腾讯实时 + westock K线 + tushare 财务 + 三大指数</div>
  <div class="concl">结论：__CONCLUSION__</div>
</div>
__SNAP__
__CHARTS__
<div class="section"><h2>📊 技术面（__STRENGTH__）</h2>__TECH____COPY_TECH__</div>
<div class="section"><h2>🏦 基本面</h2>__FUND____COPY_FUND__</div>
<div class="section"><h2>🏭 板块与概念</h2>
  <p><b>所属行业：</b>__SECTOR__</p>
  <p><b>主营业务：</b>__BUSINESS__</p>
  __COPY_SECTOR__
</div>
<div class="section"><h2>🌐 大盘环境对比（__MKT_ENV__）</h2>__MKT____COPY_MKT__</div>
<div class="section"><h2>⚠️ 风险提示</h2><ul class="risk">__RISK__</ul></div>
<div class="section"><h2>💡 操作建议</h2><ul class="advice">__ADVICE__</ul>__COPY_ADVICE__</div>
<div class="section"><p style="font-size:12px;color:#94a3b8">⚠️ 免责声明：本报告由程序基于公开行情数据自动生成，仅供研究参考，不构成投资建议。实时数据请以券商终端为准。</p></div>
<script>
function copyCodes(btn){
  var codes=btn.getAttribute('data-codes');
  function ok(b){var t=b.textContent;b.textContent='已复制 ✓';setTimeout(function(){b.textContent=t;},1500);}
  function fb(b,c){var ta=document.createElement('textarea');ta.value=c;document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}catch(e){}document.body.removeChild(ta);ok(b);}
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(codes).then(function(){ok(btn);},function(){fb(btn,codes);});}
  else{fb(btn,codes);}
}
</script>
</body>
</html>
"""
    html = (html.replace("__CODE__", code).replace("__NAME__", name).replace("__NOW__", now)
            .replace("__CONCLUSION__", conclusion).replace("__SNAP__", snap).replace("__CHARTS__", charts)
            .replace("__STRENGTH__", strength).replace("__TECH__", tech_tbl).replace("__FUND__", fund_tbl)
            .replace("__SECTOR__", sector).replace("__BUSINESS__", business).replace("__MKT_ENV__", mkt_txt)
            .replace("__MKT__", mkt_tbl).replace("__RISK__", risk_html).replace("__ADVICE__", advice_html)
            .replace("__COPY_TECH__", _codes_block_html(code)).replace("__COPY_FUND__", _codes_block_html(code))
            .replace("__COPY_SECTOR__", _codes_block_html(code)).replace("__COPY_MKT__", _codes_block_html(code))
            .replace("__COPY_ADVICE__", _codes_block_html(code)))
    return html


# ============================================================
#  9. 主流程
# ============================================================
def research_one(code: str):
    code = code.strip().zfill(6)
    logger.info(f"调研 {code} ...")
    q = tencent_quote(code)
    kline = CLI.get_kline(code, days=120, adjust="qfq")
    tech = compute_technical(kline)
    fin = fetch_fundamentals(code)
    sector, business = fetch_sector(code)
    mkt = market_env()
    # 腾讯失败时回退 kline 最新收盘
    if (not q.get("price") or q.get("price") == "-") and tech:
        q["price"] = _f(tech["price"])
    md = build_md(code, q, tech, fin, sector, business, mkt)
    html = build_html(code, q, tech, fin, sector, business, mkt)
    os.makedirs(OUT_DIR, exist_ok=True)
    mdp = os.path.join(OUT_DIR, f"{code}_调研.md")
    htmlp = os.path.join(OUT_DIR, f"{code}_调研.html")
    io.open(mdp, "w", encoding="utf-8").write(md)
    io.open(htmlp, "w", encoding="utf-8").write(html)
    return mdp, htmlp


def main():
    ap = argparse.ArgumentParser(description="个股深度调研报告生成器")
    ap.add_argument("codes", nargs="+", help="股票代码，支持多只")
    args = ap.parse_args()
    for code in args.codes:
        try:
            mdp, htmlp = research_one(code)
            print(f"  ✅ {code}: {mdp}\n          {htmlp}")
        except Exception as e:
            logger.error(f"{code} 调研失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
