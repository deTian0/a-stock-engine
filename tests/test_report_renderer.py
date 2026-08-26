"""report_renderer 单元测试：格式化助手 + 渲染（正常/边界/NaN 健壮性）。

重点: _fmt_pct / _fmt_num / _fmt_amt 必须正确处理 NaN（当前实现会输出 "nan%",
这是真实 bug，见下方会失败的断言）。
"""
import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import report_renderer as rr


def test_esc_basic():
    assert rr._esc('<a>&"') == "&lt;a&gt;&amp;&quot;"


def test_esc_none():
    assert rr._esc(None) == ""


def test_fmt_pct_none():
    assert rr._fmt_pct(None) == "-"


def test_fmt_pct_positive_sign():
    assert rr._fmt_pct(3.2) == "+3.2%"


def test_fmt_pct_negative():
    assert rr._fmt_pct(-1.5) == "-1.5%"


def test_fmt_pct_nan_shows_dash():
    # BUG 捕获: 当前 _fmt_pct(np.nan) 返回 'nan%'，应返回 '-'
    assert rr._fmt_pct(np.nan) == "-"


def test_fmt_num_zero_dash():
    assert rr._fmt_num(0) == "-"
    assert rr._fmt_num(None) == "-"


def test_fmt_num_nan_shows_dash():
    # BUG 捕获: 当前 _fmt_num(np.nan) 返回 'nan'，应返回 '-'
    assert rr._fmt_num(np.nan) == "-"


def test_fmt_num_normal():
    assert rr._fmt_num(12.3456, 2) == "12.35"


def test_fmt_amt_nan_shows_dash():
    assert rr._fmt_amt(np.nan) == "-"


def test_is_garbage_n_stocks_and_st():
    assert rr._is_garbage({"name": "N新股"}) is True
    assert rr._is_garbage({"name": "ST某某"}) is True
    assert rr._is_garbage({"name": "ST"}) is True
    assert rr._is_garbage({"name": "贵州茅台"}) is False


def test_cls():
    assert rr._cls(1) == "up"
    assert rr._cls(-1) == "down"
    assert rr._cls(0) == "up"


def test_clean_top_sectors_drops_comprehensive_and_outliers():
    raw = [
        {"sector": "综合", "avg_chg": 5.0, "max_chg": 6, "amount_yi": 10},
        {"sector": "半导体", "avg_chg": 99.0, "max_chg": 100, "amount_yi": 10},  # 极值
        {"sector": "光伏", "avg_chg": 2.3, "max_chg": 3.0, "amount_yi": 8},
        {"sector": "无均值", "amount_yi": 1},  # 缺 avg_chg
    ]
    out = rr._clean_top_sectors(raw)
    sectors = [r["sector"] for r in out]
    assert "综合" not in sectors
    assert "半导体" not in sectors       # 极值保护
    assert "无均值" not in sectors
    assert sectors == ["光伏"]


def test_md_block_to_html_table():
    md = "| 代码 | 名称 |\n|------|------|\n| 600519 | 茅台 |"
    html = rr._md_block_to_html(md)
    assert "<table>" in html
    assert "茅台" in html


def test_count_t0():
    t0 = {"by_cat": {"②A_质量榜": [1, 2], "②B_短线榜": [3]}}
    assert rr._count_t0(t0) == 3
    assert rr._count_t0(None) == 0


def test_render_review_report_no_leftover_placeholders():
    data = {
        "top_sectors": [{"sector": "光伏", "avg_chg": 2.3, "max_chg": 3.0,
                         "amount_yi": 8, "stock_count": 5}],
        "hit_stats": {"total_hits": 1, "total_stocks": 2, "total_cycles": 1,
                      "active_count": 1, "pre_top5": [], "post_top5": []},
        "t0_verify": None,
        "weekly": {},
        "sector_stocks": {},
        "tracking_md": "",
    }
    html = rr.render_review_report(data)
    assert isinstance(html, str) and len(html) > 100
    assert "{{" not in html   # 防御性校验: 不应残留占位符
