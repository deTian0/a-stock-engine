"""
kline_tracker.py — 推荐标的日K线追踪报告

补齐 pick_tracker 缺失的「可视化」一环：
  pick_tracker 只做命中计数（14天周期、首次命中起算、周期内再命中→延长），
  本模块把「处于追踪周期内的早盘推荐股票 / ETF」的日K线画出来，
  让你能直观看到从选中那天起到今天的价格轨迹与盈亏。

数据源（重要）：
  - 主源 = LocalPriceLoader（westock-data CLI / KlineCache）：返回真实 OHLC 蜡烛 +
    成交量，且是早盘简报画图用的同一接口，数据近期（随每日简报运行自动累积缓存）。
  - 回退 = data_cache/market.db 的 daily_price 收盘价：仅当 westock 彻底取不到、
    且该标的在库中确有近期数据（MAX(date)>=选中日）时才用，避免拿数月前的陈旧
    收盘价误导追踪。否则诚实标「无数据」。
  （注：本地 market.db 对绝大多数个股已陈旧至 2026-03/04，故不能作主源。）

追踪周期逻辑（复用 pick_tracker 的 pick_summary）：
  - active_cycle_end >= 今日 即视为「追踪中」
  - active_cycle_hits >= 2 即「🔁 已延长」（期间再次被推荐，周期顺延）

用法：
    python -m src.kline_tracker                       # 默认：今日 as_of，盘前，全部活跃
    python -m src.kline_tracker --date 2026-08-13     # 指定截止日
    python -m src.kline_tracker --session post_market # 盘后
    python -m src.kline_tracker --limit 30            # 仅取周期命中最高的前 N 只
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pick_tracker import PickTracker

logger = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
MARKET_DB = REPO / "data_cache" / "market.db"
KLINE_DIR = REPO / "data_cache" / "kline"
TRACK_DB = REPO / "history" / "picks.db"


# ================================================================
#  数据获取
# ================================================================
def normalize_code(code: str) -> str:
    """转成 6 位裸码。"""
    return str(code).strip().split(".")[0].zfill(6)


def _market_candidates(code: str) -> list:
    """daily_price 里代码格式混杂（600000.SH / 510010 裸 / 000001.SZ），构造候选集。"""
    bare = normalize_code(code)
    return [bare, f"{bare}.SH", f"{bare}.SZ"]


def fetch_close_series(code: str, start: str, end: str) -> pd.DataFrame:
    """从 market.db 取收盘价序列（date, close, vol, amount），按日期升序。"""
    if not MARKET_DB.exists():
        return pd.DataFrame()
    try:
        import sqlite3
        cands = _market_candidates(code)
        qmarks = ",".join("?" * len(cands))
        con = sqlite3.connect(str(MARKET_DB))
        df = pd.read_sql_query(
            f"SELECT date, close, vol, amount FROM daily_price "
            f"WHERE code IN ({qmarks}) AND date>=? AND date<=? ORDER BY date ASC",
            con, params=cands + [start, end],
        )
        con.close()
        if len(df) == 0:
            return df
        df["close"] = df["close"].astype(float)
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        logger.warning(f"取收盘价失败 {code}: {e}")
        return pd.DataFrame()


def fetch_cache_ohlc(code: str, start: str, end: str) -> dict:
    """保留：从 westock KlineCache JSON 直接取真实 OHLC（LocalPriceLoader 内部已用，
    这里仅在需要逐日精确标记时备用）。"""
    bare = normalize_code(code)
    fp = KLINE_DIR / f"{bare}_daily.json"
    if not fp.exists():
        return {}
    try:
        arr = json.loads(fp.read_text(encoding="utf-8"))
        out = {}
        for r in arr:
            d = r.get("date")
            if not d or d < start or d > end:
                continue
            try:
                out[d] = (float(r["open"]), float(r["close"]),
                          float(r["low"]), float(r["high"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out
    except Exception as e:
        logger.debug(f"读 KlineCache 失败 {code}: {e}")
        return {}


_LOADER = None


def _get_loader():
    global _LOADER
    if _LOADER is None:
        from local_price_loader import LocalPriceLoader
        _LOADER = LocalPriceLoader()
    return _LOADER


def _code_variants(code: str) -> list:
    """westock 对不同品种的代码格式敏感，尝试多种写法提升命中。"""
    bare = normalize_code(code)
    return [bare, "sh" + bare, "sz" + bare, bare + ".SH", bare + ".SZ"]


def build_kline(code: str, start: str, end: str) -> dict:
    """
    返回 {dates, ohlc:[[o,c,l,h]...], vol:[...], ohlc_source}。
    主源 = LocalPriceLoader（westock 真实 OHLC，近期）；
    回退 = market.db 收盘价（仅当 westock 取不到且该标的库内确有近期数据）；
    否则返回 {}（无数据）。
    """
    bare = normalize_code(code)
    # --- 主源：westock 真实蜡烛 ---
    try:
        need_days = max(30, (datetime.strptime(end, "%Y-%m-%d") -
                             datetime.strptime(start, "%Y-%m-%d")).days + 3)
        df = None
        for cand in _code_variants(code):
            df = _get_loader().get_price(cand, days=need_days)
            if df is not None and len(df) > 0:
                break
        if df is not None and len(df) > 0:
            df = df.copy()
            df["date"] = df["date"].astype(str)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) >= 1:
                opens = pd.to_numeric(df["open"], errors="coerce")
                closes = pd.to_numeric(df["close"], errors="coerce")
                lows = pd.to_numeric(df["low"], errors="coerce")
                highs = pd.to_numeric(df["high"], errors="coerce")
                volcol = "volume" if "volume" in df.columns else "vol"
                vols = pd.to_numeric(df[volcol], errors="coerce").fillna(0.0)
                ohlc = [[round(float(o), 3), round(float(c), 3),
                         round(float(l), 3), round(float(h), 3)]
                        for o, c, l, h in zip(opens, closes, lows, highs)]
                return {
                    "dates": df["date"].tolist(),
                    "ohlc": ohlc,
                    "vol": [round(float(v), 2) for v in vols],
                    "ohlc_source": "westock",
                }
    except Exception as e:
        logger.debug(f"westock 取 {code} 失败: {e}")

    # --- 回退：market.db 收盘价（仅当确有近期数据）---
    base = fetch_close_series(code, start, end)
    if base is not None and len(base) > 0:
        dates = base["date"].tolist()
        closes = base["close"].values
        vols = base["vol"].values
        ohlc = []
        for i, d in enumerate(dates):
            c = closes[i]
            o = closes[i - 1] if i > 0 else c
            hi = max(o, c)
            lo = min(o, c)
            ohlc.append([round(o, 3), round(c, 3), round(lo, 3), round(hi, 3)])
        return {
            "dates": dates,
            "ohlc": ohlc,
            "vol": [round(float(v), 2) for v in vols],
            "ohlc_source": "reconstructed",
        }
    return {}


# ================================================================
#  活跃追踪周期
# ================================================================
def get_active_tracked(session_type: str = "pre_market", as_of: str = None,
                       limit: int = 0) -> list:
    """
    读取 pick_summary 中「追踪中」的标的。
    返回 list[dict]: code,name,cycle_start,cycle_end,cycle_hits,cumulative_hits,
                     first_pick,last_pick,extended(bool)
    """
    if as_of is None:
        as_of = datetime.now().strftime("%Y-%m-%d")
    tr = PickTracker()
    c = tr.db.conn
    rows = c.execute(
        "SELECT code,name,active_cycle_start,active_cycle_end,active_cycle_hits,"
        "cumulative_hits,first_pick_date,last_pick_date "
        "FROM pick_summary WHERE session_type=? AND active_cycle_end>=? "
        "ORDER BY active_cycle_hits DESC, cumulative_hits DESC, code ASC",
        (session_type, as_of),
    ).fetchall()
    out = []
    for r in rows:
        cycle_hits = r["active_cycle_hits"] or 1
        out.append({
            "code": r["code"],
            "name": r["name"] or r["code"],
            "cycle_start": r["active_cycle_start"],
            "cycle_end": r["active_cycle_end"],
            "cycle_hits": cycle_hits,
            "cumulative_hits": r["cumulative_hits"] or 1,
            "first_pick": r["first_pick_date"],
            "last_pick": r["last_pick_date"],
            "extended": cycle_hits >= 2,
        })
    if limit and limit > 0:
        out = out[:limit]
    return out


# ================================================================
#  HTML 报告
# ================================================================
def _compute_return(kl: dict, cycle_start: str) -> float:
    """区间收益：从 cycle_start（或首根可用）到末根的收盘价涨跌幅。"""
    if not kl or not kl.get("dates"):
        return float("nan")
    dates = kl["dates"]
    closes = [row[1] for row in kl["ohlc"]]  # close 在 ohlc[1]
    # 找 cycle_start 之后（含）的第一根
    idx = 0
    for i, d in enumerate(dates):
        if d >= cycle_start:
            idx = i
            break
    if idx >= len(closes) or len(closes) < 2:
        return float("nan")
    p0, p1 = closes[idx], closes[-1]
    if p0 <= 0:
        return float("nan")
    return round((p1 / p0 - 1) * 100, 2)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>推荐标的日K线追踪 — {as_of}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
:root{{--bg:#0e1117;--card:#161b22;--fg:#e6edf3;--mut:#8b949e;--acc:#58a6ff;--up:#ef4444;--dn:#22c55e;--warn:#f59e0b;}}
*{{box-sizing:border-box;}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;}}
header{{padding:18px 24px;border-bottom:1px solid #21262d;}}
header h1{{margin:0 0 4px;font-size:20px;}}
header .sub{{color:var(--mut);font-size:13px;}}
.wrap{{padding:16px 24px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;}}
th,td{{padding:7px 10px;border-bottom:1px solid #21262d;text-align:right;}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left;}}
th{{color:var(--mut);font-weight:600;}}
.pos{{color:var(--up);}} .neg{{color:var(--dn);}} .ext{{color:var(--warn);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px;}}
.card{{background:var(--card);border:1px solid #21262d;border-radius:10px;padding:10px 12px;}}
.card .hd{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}}
.card .hd .nm{{font-weight:600;font-size:15px;}}
.card .hd .meta{{font-size:11px;color:var(--mut);}}
.chart{{width:100%;height:240px;}}
.legend{{font-size:11px;color:var(--mut);margin-top:4px;}}
</style>
</head>
<body>
<header>
  <h1>📈 推荐标的日K线追踪</h1>
  <div class="sub">截止 {as_of} · 会话 {session_label} · 追踪中 {n_active} 只（含延长期）·
  生成 {gen_time} · 红涨绿跌</div>
</header>
<div class="wrap">
  <table>
    <thead><tr>
      <th>代码</th><th>名称</th><th>选中日</th><th>持仓天数</th>
      <th>区间收益</th><th>周期内命中</th><th>累计命中</th><th>状态</th><th>K线来源</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="grid" id="grid"></div>
</div>
<script>
const DATA = {data_json};
const SRC_LABEL = {{westock:"真实蜡烛(westock)",reconstructed:"收盘价重建(库陈旧)",none:"无数据"}};
const grid = document.getElementById('grid');
DATA.forEach(function(it, i){{
  const div = document.createElement('div');
  div.className = 'card';
  const retCls = it.ret>=0 ? 'pos':'neg';
  const retTxt = (it.ret>=0?'+':'') + it.ret.toFixed(2) + '%';
  const status = it.extended ? '<span class="ext">🔁 已延长</span>' : '追踪中';
  const winEnd = it.window_end;
  div.innerHTML = '<div class="hd"><div class="nm">'+it.name+
    ' <span style="color:#8b949e;font-size:12px">'+it.code+'</span></div>'+
    '<div class="meta">'+it.cycle_start+' 起 · '+it.hold_days+'天 · <b class="'+retCls+'">'+retTxt+'</b></div></div>'+
    '<div class="chart" id="c'+i+'"></div>'+
    '<div class="legend">'+status+' · K线: '+SRC_LABEL[it.ohlc_source]+'</div>';
  grid.appendChild(div);
  const chart = echarts.init(document.getElementById('c'+i), 'dark');
  const kl = it.ohlc;            // [o,c,l,h]
  const dates = it.dates;
  const vol = it.vol;
  const candle = kl.map(function(r){{return [r[0],r[1],r[2],r[3]];}});
  const volData = vol.map(function(v,idx){{return {{value:v, itemStyle:{{color: kl[idx][1]>=kl[idx][0]?'#ef4444':'#22c55e'}}}};}});
  const opt = {{
    backgroundColor:'transparent',
    grid:[{{left:48,right:12,top:10,height:'62%'}},{{left:48,right:12,top:'76%',height:'16%'}}],
    xAxis:[
      {{type:'category',data:dates,axisLabel:{{show:false}},axisLine:{{lineStyle:{{color:'#30363d'}}}}}},
      {{type:'category',data:dates,gridIndex:1,axisLabel:{{show:false}},axisLine:{{lineStyle:{{color:'#30363d'}}}}}}
    ],
    yAxis:[
      {{scale:true,splitLine:{{lineStyle:{{color:'#21262d'}}}},axisLabel:{{color:'#8b949e'}}}},
      {{scale:true,gridIndex:1,splitLine:{{show:false}},axisLabel:{{show:false}}}}
    ],
    tooltip:{{trigger:'axis',axisPointer:{{type:'cross'}}}},
    axisPointer:{{link:[{{xAxisIndex:'all'}}]}},
    series:[
      {{
        type:'candlestick',data:candle,
        itemStyle:{{color:'#ef4444',color0:'#22c55e',borderColor:'#ef4444',borderColor0:'#22c55e'}},
        markLine:{{symbol:'none',silent:true,lineStyle:{{color:'#58a6ff',type:'dashed'}},
          data:[{{xAxis:it.cycle_start,label:{{formatter:'选中',color:'#58a6ff',fontSize:10}}}}]}},
        markArea:{{silent:true,itemStyle:{{color:'rgba(88,166,255,0.06)'}},
          data:[[{{xAxis:it.cycle_start}},{{xAxis:winEnd}}]]}}
      }},
      {{type:'bar',data:volData,xAxisIndex:1,yAxisIndex:1}}
    ]
  }};
  chart.setOption(opt);
  window.addEventListener('resize', function(){{chart.resize();}});
}});
</script>
</body>
</html>
"""


def build_html(items: list, as_of: str, session_label: str) -> str:
    """生成自包含 HTML 报告。"""
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 汇总表行
    rows_html = []
    data = []
    for it in items:
        kl = build_kline(it["code"], it["cycle_start"], as_of)
        if not kl:
            # 无数据：仍列入表格但图表空缺
            ret = float("nan")
            src = "无数据"
            window_end = it["cycle_end"]
            hold_days = max(0, (datetime.strptime(min(as_of, it["cycle_end"]), "%Y-%m-%d") -
                                 datetime.strptime(it["cycle_start"], "%Y-%m-%d")).days)
            data.append({
                "code": it["code"], "name": it["name"], "cycle_start": it["cycle_start"],
                "window_end": window_end, "hold_days": hold_days, "ret": 0,
                "dates": [], "ohlc": [], "vol": [], "ohlc_source": "none",
                "extended": it["extended"],
            })
            ret_cls, ret_txt = "", "—"
        else:
            ret = _compute_return(kl, it["cycle_start"])
            src = kl["ohlc_source"]
            window_end = min(as_of, it["cycle_end"])
            hold_days = max(0, (datetime.strptime(window_end, "%Y-%m-%d") -
                                 datetime.strptime(it["cycle_start"], "%Y-%m-%d")).days)
            data.append({
                "code": it["code"], "name": it["name"], "cycle_start": it["cycle_start"],
                "window_end": window_end, "hold_days": hold_days, "ret": (ret if pd.notna(ret) else 0),
                "dates": kl["dates"], "ohlc": kl["ohlc"], "vol": kl["vol"],
                "ohlc_source": src, "extended": it["extended"],
            })
            if pd.notna(ret):
                ret_cls = "pos" if ret >= 0 else "neg"
                ret_txt = f"{ret:+.2f}%"
            else:
                ret_cls, ret_txt = "", "—"

        ext_cls = "ext" if it["extended"] else ""
        ext_txt = "🔁 已延长" if it["extended"] else "追踪中"
        rows_html.append(
            f"<tr><td>{it['code']}</td><td>{it['name']}</td>"
            f"<td>{it['cycle_start']}</td><td>{hold_days}</td>"
            f"<td class='{ret_cls}'>{ret_txt}</td>"
            f"<td>{it['cycle_hits']}</td><td>{it['cumulative_hits']}</td>"
            f"<td class='{ext_cls}'>{ext_txt}</td><td>{src}</td></tr>"
        )

    html = HTML_TEMPLATE.format(
        as_of=as_of,
        session_label=session_label,
        n_active=len(items),
        gen_time=gen_time,
        rows="".join(rows_html),
        data_json=json.dumps(data, ensure_ascii=False),
    )
    return html


# ================================================================
#  CLI
# ================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="推荐标的日K线追踪报告")
    ap.add_argument("--date", help="截止日 YYYY-MM-DD（默认今天）")
    ap.add_argument("--session", default="pre_market",
                    choices=["pre_market", "post_market"])
    ap.add_argument("--limit", type=int, default=0, help="仅取周期命中最高的前 N 只（0=全部）")
    ap.add_argument("--out", help="输出 HTML 路径")
    args = ap.parse_args()

    as_of = args.date or datetime.now().strftime("%Y-%m-%d")
    session_label = "盘前" if args.session == "pre_market" else "盘后"

    items = get_active_tracked(args.session, as_of, limit=args.limit)
    if not items:
        print(f"无追踪中的标的（session={args.session}, as_of={as_of}）")
        return

    html = build_html(items, as_of, session_label)

    out = args.out or str(REPO / "history" / as_of / "追踪_日K图.html")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    # 固定名指针（最新一份）
    pointer = REPO / "history" / "追踪_日K图_latest.html"
    pointer.write_text(html, encoding="utf-8")

    extended = [it for it in items if it["extended"]]
    print(f"追踪报告已生成: {out_path}")
    print(f"追踪中 {len(items)} 只 | 已延长(再命中) {len(extended)} 只")
    if extended:
        print("🔁 已延长期（期间再次命中）:",
              ", ".join(f"{it['name']}({it['code']})×{it['cycle_hits']}" for it in extended[:10]))


if __name__ == "__main__":
    main()
