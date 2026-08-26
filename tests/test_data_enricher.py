"""data_enricher 单元测试：纯函数 + 全字段 df 的 enrich_and_report（无网络）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import data_enricher as de


def test_grade_signal_fire3():
    row = {"composite_score": 95, "tech_signal": "偏多", "roe": 20}
    assert de.grade_signal(row) == "🔥🔥🔥"


def test_grade_signal_neutral():
    row = {"composite_score": 40, "tech_signal": "震荡", "roe": 0}
    assert de.grade_signal(row) == "⚪"


def test_grade_signal_bearish_penalty():
    row = {"composite_score": 80, "tech_signal": "偏空", "roe": 5}
    # 高分但偏空 -> 不应给最高级
    assert de.grade_signal(row) != "🔥🔥🔥"


def test_infer_sector():
    assert de._infer_sector("600519") == "沪市"
    assert de._infer_sector("300750") == "创业板"
    assert de._infer_sector("515790") == "ETF沪"
    assert de._infer_sector("159852") == "ETF深"


def _full_df():
    return pd.DataFrame({
        "code": ["600519", "000001"],
        "name": ["茅台", "平安"],
        "close": [1800.0, 15.0],
        "concept_name": ["白酒", "银行"],
        "concept_chg": [1.2, -0.5],
        "tech_signal": ["偏多", "偏空"],
        "tech_ma": ["MA20>MA60", "MA20<MA60"],
        "tech_macd": ["金叉", "死叉"],
        "tech_rsi": ["60", "40"],
        "sector": ["白酒", "银行"],
        "composite_score": [85.0, 60.0],
    })


def test_enrich_and_report_no_network():
    df = _full_df()
    out = de.enrich_and_report(df)
    # 全部已填充 -> 不触发任何网络补全, 仅加 signal_grade
    assert "signal_grade" in out.columns
    assert out["signal_grade"].notna().all()
    # 原始列保留
    assert "close" in out.columns
    assert out["close"].iloc[0] == 1800.0
