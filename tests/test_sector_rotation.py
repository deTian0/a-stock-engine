"""sector_rotation_watchlist 单元测试：强度计算 / 轮动识别 / 报告（正常/边界）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sector_rotation_watchlist import SectorRotationWatcher


def _sector_df():
    return pd.DataFrame({
        "name": ["半导体", "银行", "白酒", "医药"],
        "change_5d": [5.0, -3.0, 1.0, 0.0],
        "change_20d": [8.0, -5.0, 2.0, 0.5],
        "amount_change": [0.5, -0.5, 0.1, 0.0],
    })


def test_calc_sector_strength_sorted_desc():
    w = SectorRotationWatcher.__new__(SectorRotationWatcher)
    out = w.calc_sector_strength(_sector_df())
    assert "strength_score" in out.columns
    vals = out["strength_score"].tolist()
    assert vals == sorted(vals, reverse=True)
    assert out.iloc[0]["name"] == "半导体"  # 5d+20d 最强


def test_calc_sector_strength_empty():
    w = SectorRotationWatcher.__new__(SectorRotationWatcher)
    df = pd.DataFrame()
    assert len(w.calc_sector_strength(df)) == 0


def test_identify_rotation_classifies():
    w = SectorRotationWatcher.__new__(SectorRotationWatcher)
    df = w.calc_sector_strength(_sector_df())
    rot = w.identify_rotation(df)
    assert "半导体" in rot["inflow"]
    assert "银行" in rot["outflow"]
    assert rot["rotation"] == []  # 需历史对比, 默认空


def test_generate_report_contains_sections():
    w = SectorRotationWatcher.__new__(SectorRotationWatcher)
    df = w.calc_sector_strength(_sector_df())
    rot = w.identify_rotation(df)
    rep = w.generate_report(df, rot)
    assert "板块轮动监控" in rep
    assert "资金流入板块" in rep
    assert "资金流出板块" in rep
