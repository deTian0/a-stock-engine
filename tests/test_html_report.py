"""html_report 单元测试：格式化助手 + generate_html（正常/边界）+ 一键复制代码块。"""
import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import html_report
from html_report import generate_html


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


def _build_results():
    quality = pd.DataFrame([
        {"code": "600519", "name": "茅台", "composite_score": 90.0, "roe": 25.0,
         "close": 1800.0, "stop_loss": 1750.0, "tech_signal": "偏多",
         "tech_ma": "站上MA20,多头", "tech_macd": "金叉", "liquidity_tag": "正常",
         "signal_grade": "⚪", "sector": "白酒", "momentum_20d": 3.0},
        {"code": "000001", "name": "平安", "composite_score": 80.0, "roe": 12.0,
         "close": 15.0, "stop_loss": 14.5, "tech_signal": "震荡",
         "tech_ma": "跌破MA20,空头", "tech_macd": "死叉", "liquidity_tag": "正常",
         "signal_grade": "⚪", "sector": "银行", "momentum_20d": 2.0},
    ])
    short = pd.DataFrame([
        {"code": "601318", "name": "平安2", "composite_score": 82.0, "roe": 14.0,
         "close": 55.0, "stop_loss": 53.0, "tech_signal": "偏多",
         "tech_ma": "站上MA20,多头", "tech_macd": "金叉", "liquidity_tag": "正常",
         "signal_grade": "⚪", "sector": "保险", "concept_name": "保险",
         "concept_chg": 1.0, "momentum_20d": 2.0},
    ])
    etf = pd.DataFrame([
        {"code": "512480", "name": "半导体ETF", "etf_type": "宽基", "momentum_20d": 5.0,
         "amount": 1e9, "score": 80, "advice": "定投"},
        {"code": "515050", "name": "5GETF", "etf_type": "宽基", "momentum_20d": 4.0,
         "amount": 1e9, "score": 80, "advice": "定投"},
    ])
    return {
        "timestamp": "t", "elapsed_seconds": 1.0,
        "regime": {"regime": "震荡", "position_cap": 0.5, "indices": {}},
        "categories": {"②A_质量榜": quality, "②B_短线榜": short},
        "etf_picks": etf,
        "l4_results": quality,
    }


def test_html_has_thsx_copy_block(config):
    """新增功能: HTML 每个板块下须有一键复制同花顺代码块(含复制按钮 + data-codes)。"""
    res = _build_results()
    html = generate_html(res, config)
    assert "同花顺自选" in html
    assert "copyCodes(this)" in html, "缺少复制按钮 onclick 绑定"
    assert "data-codes=" in html, "复制块缺少 data-codes 属性"
    # 复制块须包含真实代码(中长线 + ETF)
    assert "600519" in html and "512480" in html
    # 复制按钮文案
    assert "复制全部" in html


def test_html_copy_block_once_per_section(config):
    """回归: HTML 复制块每节仅一个(中长线/短线/ETF 各一), 不重复。"""
    res = _build_results()
    html = generate_html(res, config)
    # 3 个板块(②A + ②B + ETF), 每板块代码 <=5 -> 各 1 个 block
    assert html.count("同花顺自选（一键复制）") == 3, "HTML 复制块数量异常"
