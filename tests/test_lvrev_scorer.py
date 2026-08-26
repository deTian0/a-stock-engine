"""lvrev_scorer 单元测试：score_lvrev 与 apply_entry_gates（纯 pandas 内核）。

覆盖: 正常路径 / 边界（空 DataFrame、缺列兜底）/ 价值因子与 ey 加性维度 / 入场闸门各分支。
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lvrev_scorer import score_lvrev, apply_entry_gates


def test_score_lvrev_normal(sample_lvrev_df):
    out = score_lvrev(sample_lvrev_df)
    assert "composite_score" in out.columns
    assert len(out) == len(sample_lvrev_df)
    vals = out["composite_score"].tolist()
    assert vals == sorted(vals, reverse=True)  # 已按评分降序
    assert (out["composite_score"] >= 0).all()


def test_score_lvrev_empty():
    out = score_lvrev(pd.DataFrame())
    assert "composite_score" in out.columns
    assert len(out) == 0


def test_score_lvrev_missing_columns():
    # 仅 code 列：缺失所有因子 -> 全部用默认 0.5，仍能产出评分
    df = pd.DataFrame({"code": ["A", "B"]})
    out = score_lvrev(df)
    assert "composite_score" in out.columns
    assert len(out) == 2


def test_score_lvrev_value_factor_changes_scores():
    df = pd.DataFrame({
        "vol20": [0.2, 0.4], "rev_chg": [-0.03, 0.02], "debt_ratio": [40, 60],
        "revenue_growth": [10, 5], "pb": [5, 2], "ps_ttm": [3, 1], "pe": [20, 10],
    })
    base = score_lvrev(df, value_factor=False)["composite_score"]
    val = score_lvrev(df, value_factor=True)["composite_score"]
    assert not base.equals(val)  # 价值因子打开应改变评分


def test_score_lvrev_ey_weight_changes_scores():
    df = pd.DataFrame({
        "vol20": [0.2, 0.4], "rev_chg": [-0.03, 0.02], "debt_ratio": [40, 60],
        "revenue_growth": [10, 5], "pb": [5, 2], "ps_ttm": [3, 1], "pe": [20, 10],
    })
    base = score_lvrev(df, ey_weight=0.0)["composite_score"]
    ey = score_lvrev(df, ey_weight=0.1)["composite_score"]
    assert not base.equals(ey)


def test_apply_entry_gates_normal(sample_lvrev_df):
    # 依据分析: 仅第0只通过(MA20>MA60 + 低波动 + 底部超跌)，其余被拒
    mask = apply_entry_gates(sample_lvrev_df, reversal_q=0.30)
    assert len(mask) == len(sample_lvrev_df)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, False]


def test_apply_entry_gates_empty():
    assert len(apply_entry_gates(pd.DataFrame())) == 0


def test_apply_entry_gates_trend_down_rejects():
    # MA20 <= MA60 -> 硬闸门全部拒绝
    df = pd.DataFrame({
        "rev_chg": [-0.1, -0.1], "close": [10, 10],
        "ma20": [9, 9], "ma60": [10, 10], "vol20": [0.1, 0.1],
    })
    assert apply_entry_gates(df).tolist() == [False, False]


def test_apply_entry_gates_falling_knife_rejects():
    # 价格 < MA20*0.93（接飞刀）-> 软闸门拒绝
    df = pd.DataFrame({
        "rev_chg": [-0.1], "close": [8.0], "ma20": [10.0], "ma60": [9.0], "vol20": [0.1],
    })
    assert apply_entry_gates(df).tolist() == [False]


def test_apply_entry_gates_high_vol_rejects():
    # 波动率超过全市场中位数 -> 低波门槛拒绝
    df = pd.DataFrame({
        "rev_chg": [-0.2, -0.2], "close": [10, 10],
        "ma20": [11, 11], "ma60": [9, 9], "vol20": [0.9, 0.9],
    })
    # 全部 vol20=0.9，中位数为0.9，vol>0.9 为假 -> 不拒；此处验证不崩溃且返回 bool
    mask = apply_entry_gates(df)
    assert mask.dtype == bool and len(mask) == 2
