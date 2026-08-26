"""html_report 单元测试：格式化助手 + generate_html（正常/边界）。"""
import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import html_report


def test_clean():
    assert html_report._clean(None) == "-"
    assert html_report._clean("") == "-"
    assert html_report._clean("nan") == "-"
    assert html_report._clean("茅台") == "茅台"


def test_fmt_val_none_and_zero_dash():
    assert html_report._fmt_val(None) == "-"
    assert html_report._fmt_val(0) == "-"
    assert html_report._fmt_val(np.nan) == "-"


def test_fmt_val_pct():
    assert html_report._fmt_val(3.2, pct=True) == "+3.2%"
    assert html_report._fmt_val(-1.5, pct=True) == "-1.5%"


def test_fmt_val_normal_2f():
    assert html_report._fmt_val(12.345) == "12.35"


def test_generate_html_empty_results(config):
    results = {
        "timestamp": "2026-08-26 09:00",
        "elapsed_seconds": 0,
        "regime": {"regime": "空头", "position_cap": 0.20},
        "categories": {},
        "l4_results": pd.DataFrame(),
        "etf_picks": pd.DataFrame(),
    }
    html = html_report.generate_html(results, config)
    assert isinstance(html, str)
    assert "盘前选股简报" in html
    assert "空头" in html
    assert "<!DOCTYPE html>" in html


def test_generate_html_with_quality(config):
    q = pd.DataFrame({
        "code": ["600519", "000001"],
        "name": ["茅台", "平安"],
        "close": [1800.0, 15.0],
        "stop_loss": [1700.0, 14.0],
        "tech_signal": ["偏多", "偏空"],
        "tech_ma": ["MA20>MA60", "MA20<MA60"],
        "tech_macd": ["金叉", "死叉"],
        "roe": [25.0, 8.0],
        "composite_score": [85.0, 60.0],
        "liquidity_tag": ["高", "中"],
        "signal_grade": ["🔥🔥", "🔥"],
        "sector": ["白酒", "银行"],
    })
    results = {
        "timestamp": "2026-08-26 09:00",
        "elapsed_seconds": 1,
        "regime": {"regime": "多头", "position_cap": 0.80},
        "categories": {"②A_质量榜": q},
        "l4_results": q,
        "etf_picks": pd.DataFrame(),
    }
    html = html_report.generate_html(results, config)
    assert "茅台" in html
    assert "多头" in html
