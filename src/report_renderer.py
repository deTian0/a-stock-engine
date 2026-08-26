"""
report_renderer.py — 盘后复盘报告「模板 + 数据填充」渲染器。

设计原则
--------
- 表达与数据彻底分离：视觉结构 / 样式全部在 ``templates/review_report.html``，
  本模块只负责把 ``afternoon_review.review_sectors()`` 产出的结构化 data
  清洗、格式化后，填入模板的占位符。模板不再依赖任何 f-string，
  因此彻底规避了旧版 ``_HTML_CSS`` 抽成普通字符串后 f-string 插值不塌缩导致的
  CSS 双花括号失效问题。

- 修复旧版可读性问题的落点：
    * CSS 双花括号失效           -> 模板是独立文件，浏览器原生解析单花括号
    * 板块被「综合」兜底 + 新股/ST 极值污染 -> 这里直接剔除「综合」、N 字头新股、ST/*ST
    * 成交额 / 涨幅小数未舍入     -> 统一在填充层格式化 (.1f / .2f)
    * 跌价不显示负号             -> _fmt_pct 统一带符号 (+/-)
    * 第五部分标题重复 + 四/五重叠 -> 模板结构定死：四=概览+Top5，五=追踪明细(去首行H1)

占位符约定
----------
- 标量： ``{{ key }}``
- 列表循环： ``<!-- for list_name -->`` ... 行模板(可含 ``{{item_key}}``) ... ``<!-- endfor -->``
- 复杂子块（板块个股、早盘验证、追踪明细）由渲染器预拼为 HTML 注入 ``{{ xxx_html }}``，
  保持模板以"展示结构"为主、数据组装在程序侧。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "review_report.html"

_OUTLIER_CAP = 25.0  # 板块平均涨幅超过该值（±）视为新股/数据异常，剔除


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _esc(val) -> str:
    """HTML 转义。"""
    s = "" if val is None else str(val)
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt_pct(v) -> str:
    """带符号百分比；None/NaN/缺失显示 '-'（涨 + 跌 -，符合 A 股红涨绿跌观感）。"""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:  # NaN
        return "-"
    return f"{f:+.1f}%"


def _cls(v) -> str:
    """涨跌幅颜色类：涨=up(红)，跌=down(绿)。"""
    return "up" if (v or 0) >= 0 else "down"


def _fmt_num(v, nd: int = 2) -> str:
    """普通数值格式化；0 / None / NaN 显示 '-'。"""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:  # NaN
        return "-"
    if f == 0:
        return "-"
    return f"{f:.{nd}f}"


def _fmt_amt(v) -> str:
    """成交额(亿)格式化，1 位小数；None / NaN 显示 '-'。"""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:  # NaN
        return "-"
    return f"{f:.1f}"


def _is_garbage(s: dict) -> bool:
    """剔除会污染板块统计的个股：新股(N 字头，首日无涨跌幅) 与 ST/*ST。"""
    name = str(s.get("name", "") or "")
    if name.startswith("N"):
        return True
    if "ST" in name.upper():
        return True
    return False


# --------------------------------------------------------------------------- #
# Markdown -> HTML（仅命中追踪部分复用，去掉首行 H1 避免与章节标题重复）
# --------------------------------------------------------------------------- #
def _md_block_to_html(md: str) -> str:
    if not md:
        return ""
    lines = md.splitlines()
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            cls = f"track-h{lvl}" if lvl >= 2 else ""
            out.append(f"<h{lvl} class='{cls}'>{_esc(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if line.startswith("|"):
            tbl = []
            while i < n and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            rows = []
            for r in tbl:
                if re.match(r"^\|[\s:|\-]+\|$", r):  # 分隔行
                    continue
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                rows.append(cells)
            if len(rows) >= 1:
                html = "<table><tr>" + "".join(f"<th>{_esc(c)}</th>" for c in rows[0]) + "</tr>"
                for r in rows[1:]:
                    tds = "".join(f"<td>{_esc(c)}</td>" for c in r)
                    html += f"<tr>{tds}</tr>"
                html += "</table>"
                out.append(html)
            continue
        if re.match(r"^[-*]\s+", line):
            out.append("<ul>")
            while i < n and re.match(r"^[-*]\s+", lines[i].lstrip()):
                item = re.sub(r"^[-*]\s+", "", lines[i].strip())
                out.append(f"<li>{_esc(item)}</li>")
                i += 1
            out.append("</ul>")
            continue
        out.append(f"<p>{_esc(line)}</p>")
        i += 1
    return "\n".join(out)


def _tracking_html(md: str) -> str:
    """追踪明细：去掉首行 '# 选股命中追踪'，避免与第五部分章节标题重复。"""
    if not md:
        return '<p class="muted">暂无命中追踪数据</p>'
    lines = md.splitlines()
    if lines and lines[0].lstrip().startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    if not body:
        return '<p class="muted">暂无命中追踪数据</p>'
    return _md_block_to_html(body)


# --------------------------------------------------------------------------- #
# 各区块数据清洗 + 格式化
# --------------------------------------------------------------------------- #
def _clean_top_sectors(raw_top) -> list[dict]:
    out = []
    for r in (raw_top or []):
        sec = r.get("sector")
        if sec == "综合":  # 兜底板块不参与排名
            continue
        avg = r.get("avg_chg")
        if avg is None:
            continue
        if avg > _OUTLIER_CAP or avg < -_OUTLIER_CAP:  # 极值保护
            continue
        mx = r.get("max_chg")
        amt = r.get("amount_yi")
        out.append({
            "sector": sec,
            "avg_chg": _fmt_pct(avg),
            "avg_cls": _cls(avg),
            "avg_val": round(float(avg), 1),  # 图表用数值
            "stock_count": r.get("stock_count", "-"),
            "max_chg": _fmt_pct(mx),
            "max_cls": _cls(mx),
            "amount_yi": _fmt_amt(amt),
        })
    return out[:5]


def _sector_stocks_html(raw_sec: dict) -> str:
    parts = []
    for sec, stocks in (raw_sec or {}).items():
        if sec == "综合":
            continue
        kept = [s for s in (stocks or []) if not _is_garbage(s)]
        if not kept:
            continue
        parts.append(f'<h3 class="sub">{_esc(sec)}</h3>')
        parts.append('<table><tr><th>代码</th><th>名称</th><th>股价</th><th>一手价</th>'
                     '<th>止损</th><th>涨幅</th><th>量比</th><th>成交额(亿)</th><th>动因</th></tr>')
        for s in kept:
            chg = s.get("chg")
            causes = " ".join(f"<span class='tag'>{_esc(c)}</span>" for c in (s.get("causes") or []))
            parts.append(
                f"<tr><td>{_esc(s.get('code', '-'))}</td>"
                f"<td>{_esc(s.get('name', '-'))}</td>"
                f"<td>{_fmt_num(s.get('close'))}</td>"
                f"<td>{_fmt_num(s.get('lot'), 0)}</td>"
                f"<td>{_fmt_num(s.get('stop_loss'))}</td>"
                f"<td class='{_cls(chg)}'>{_fmt_pct(chg)}</td>"
                f"<td>{_fmt_num(s.get('vol_ratio'))}</td>"
                f"<td>{_fmt_amt(s.get('amount_yi'))}</td>"
                f"<td>{causes}</td></tr>"
            )
        parts.append("</table>")
    return "\n".join(parts) if parts else '<p class="muted">暂无个股数据</p>'


def _t0_html(t0) -> str:
    if not t0:
        return '<p class="muted">暂无早盘选股记录</p>'
    html = f'<p class="note">早盘 run_id={_esc(t0.get("run_id", "-"))}, {_esc(t0.get("date", "-"))}</p>'
    by_cat = t0.get("by_cat", {}) or {}
    if not by_cat:
        return html + '<p class="muted">无验证明细</p>'
    for cat, rows in by_cat.items():
        html += f'<h3 class="sub">{_esc(cat)}</h3>'
        if not rows:
            html += '<p class="muted">-</p>'
            continue
        html += '<table><tr><th>代码</th><th>名称</th><th>选股评分</th><th>当日表现</th><th>建议</th></tr>'
        for r in rows:
            t0p = r.get("t0", "-")
            t0cls = ("up" if isinstance(t0p, str) and t0p.startswith("+")
                     else "down" if isinstance(t0p, str) and t0p.startswith("-") else "")
            advice = r.get("advice", "-") or "-"
            adv_cls = "up" if "符合" in advice else ("down" if "偏弱" in advice else "")
            html += (f"<tr><td>{_esc(r.get('code', '-'))}</td>"
                     f"<td>{_esc(r.get('name', '-'))}</td>"
                     f"<td>{_fmt_num(r.get('score'), 0)}</td>"
                     f"<td class='{t0cls}'>{_esc(t0p)}</td>"
                     f"<td class='{adv_cls}'>{_esc(advice)}</td></tr>")
        html += "</table>"
    return html


def _multi_hit_html(hit) -> str:
    if not hit or not hit.get("multi_hit_count"):
        return ""
    codes = ", ".join(str(c) for c in (hit.get("multi_hit_codes") or []))
    return (f'<li>周期内多次命中: {hit.get("multi_hit_count")} 只 (高频信号)</li>\n'
            f'<li>高频股票: {_esc(codes)}</li>')


def _count_t0(t0) -> int:
    if not t0:
        return 0
    return sum(len(v) for v in (t0.get("by_cat", {}) or {}).values())


def _weekly_html(weekly) -> str:
    """最近一周持仓追踪：每日均值表 + 个股明细表（持仓收益带符号）。"""
    if not weekly or not weekly.get("dates"):
        return '<p class="muted">暂无近一周选股记录</p>'
    s = weekly.get("summary", {}) or {}
    parts = []
    if s:
        parts.append(
            f'<p class="note">近7日推荐 <b>{s.get("total_picks","-")}</b> 只 · '
            f'已验证 <b>{s.get("verified","-")}</b> 只 · '
            f'胜率 <b class="{"" if s.get("win_rate") is None else ("up" if s.get("win_rate",0)>=0 else "down")}">'
            f'{s.get("win_rate","-")}%</b> · '
            f'平均持仓收益 <b class="{"" if s.get("avg_return") is None else ("up" if s.get("avg_return",0)>=0 else "down")}">'
            f'{_fmt_pct(s.get("avg_return")) if s.get("avg_return") is not None else "-"}</b></p>'
        )
    # 每日均值表
    parts.append('<div class="sub">每日持仓收益概览</div>')
    parts.append('<table><tr><th>日期</th><th>推荐数</th><th>已验证</th>'
                 '<th>平均持仓收益</th><th>胜率</th></tr>')
    for d in weekly.get("dates", []):
        avg = d.get("avg_return")
        wr = d.get("win_rate")
        parts.append(
            f"<tr><td>{_esc(d.get('date','-'))}</td>"
            f"<td>{d.get('picks','-')}</td>"
            f"<td>{d.get('verified','-')}</td>"
            f"<td class='{_cls(avg)}'>{_fmt_pct(avg) if avg is not None else '-'}</td>"
            f"<td class='{_cls(wr)}'>{wr if wr is not None else '-'}%</td></tr>"
        )
    parts.append("</table>")
    # 个股明细表
    picks = weekly.get("picks", []) or []
    if picks:
        parts.append('<div class="sub">推荐个股持仓明细</div>')
        parts.append('<table><tr><th>推荐日</th><th>代码</th><th>名称</th>'
                     '<th>类目</th><th>评分</th><th>持仓收益</th><th>状态</th></tr>')
        for p in picks:
            ret = p.get("return_pct")
            cls = _cls(ret)
            ret_s = _fmt_pct(ret) if ret is not None else "-"
            parts.append(
                f"<tr><td>{_esc(p.get('date','-'))}</td>"
                f"<td>{_esc(p.get('code','-'))}</td>"
                f"<td>{_esc(p.get('name','-'))}</td>"
                f"<td>{_esc(p.get('category','-'))}</td>"
                f"<td>{_fmt_num(p.get('score'), 0)}</td>"
                f"<td class='{cls}'>{ret_s}</td>"
                f"<td>{_esc(p.get('status','-'))}</td></tr>"
            )
        parts.append("</table>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 极简模板引擎：标量替换 + 单层列表循环
# --------------------------------------------------------------------------- #
_LOOP_RE = re.compile(r"<!--\s*for\s+(\w+)\s*-->(.*?)<!--\s*endfor\s*-->", re.S)
_SCALAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _sub_scalars(text: str, ctx: dict) -> str:
    def repl(m):
        val = ctx.get(m.group(1), "")
        return "" if val is None else str(val)
    return _SCALAR_RE.sub(repl, text)


def _render(template: str, ctx: dict) -> str:
    # 1) 列表循环展开（单层）
    def loop_repl(m):
        name = m.group(1)
        body = m.group(2)
        items = ctx.get(name, []) or []
        return "\n".join(_sub_scalars(body, item) for item in items)
    rendered = _LOOP_RE.sub(loop_repl, template)
    # 2) 剩余顶层标量替换
    rendered = _sub_scalars(rendered, ctx)
    return rendered


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def render_review_report(data: dict) -> str:
    """把 review_sectors() 的结构化 data 渲染为自包含 HTML 复盘报告。"""
    data = data or {}
    top = _clean_top_sectors(data.get("top_sectors"))
    hit = data.get("hit_stats") or {}
    t0 = data.get("t0_verify")
    weekly = data.get("weekly") or {}

    ctx = {
        "date": data.get("date", datetime.now().strftime("%Y-%m-%d")),
        "generated_at": data.get("generated_at", ""),
        "lead_sector": top[0]["sector"] if top else "-",
        "listed_sectors": len(top),
        "t0_count": _count_t0(t0),
        "total_hits": hit.get("total_hits", 0),
        "total_stocks": hit.get("total_stocks", 0),
        "total_cycles": hit.get("total_cycles", 0),
        "active_count": hit.get("active_count", 0),
        "multi_hit_html": _multi_hit_html(hit),
        "top_sectors": top,
        "sector_stocks_html": _sector_stocks_html(data.get("sector_stocks")),
        "t0_html": _t0_html(t0),
        "pre_top5": hit.get("pre_top5", []) or [],
        "post_top5": hit.get("post_top5", []) or [],
        "tracking_html": _tracking_html(data.get("tracking_md", "")),
        "weekly_html": _weekly_html(weekly),
        "chart_names": json.dumps([t["sector"] for t in top], ensure_ascii=False),
        "chart_vals": json.dumps([t["avg_val"] for t in top]),
        "weekly_dates": json.dumps(weekly.get("chart_dates", []) or [], ensure_ascii=False),
        "weekly_avg": json.dumps(weekly.get("chart_avg", []) or []),
        "weekly_count": json.dumps(weekly.get("chart_count", []) or []),
        "generated_at_full": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = _render(template, ctx)

    # 防御性校验：不应残留占位符
    leftover = _SCALAR_RE.findall(html)
    if leftover:
        import logging
        logging.getLogger(__name__).warning(f"模板渲染后残留占位符: {set(leftover)}")
    return html
