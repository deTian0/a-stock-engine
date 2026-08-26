"""factor_engine 单元测试：评分 / 风控闸门 / 选股 / 过滤（正常/边界/异常）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import factor_engine as fe


def _df():
    return pd.DataFrame({
        "code": ["600519", "000001", "300750", "600036"],
        "name": ["茅台", "平安", "宁德", "招行"],
        "sector": ["白酒", "银行", "新能源", "银行"],
        "chg_10d": [5.0, -2.0, 3.0, 1.0],
        "chg_25d": [8.0, -3.0, 5.0, 2.0],
        "chg_3d": [1.0, -1.0, 0.5, 0.2],
        "chg_6d": [2.0, -1.5, 1.0, 0.3],
        "pe": [30, 8, 50, 10],
        "pb": [10, 0.8, 5, 1.5],
        "market_cap": [2.0e11, 1.0e11, 1.5e11, 9.0e10],
        "amplitude": [2.0, 3.0, 4.0, 2.5],
        "turnover": [1.0, 2.0, 3.0, 1.5],
        "roe": [25, 8, 12, 15],
        "rs20": [2.0, -3.0, 1.0, 0.5],
        "trend_up": [True, False, True, True],
    })


def test_score_stocks_returns_composite_and_sorted_desc():
    s = fe.score_stocks(_df())
    assert "composite_score" in s.columns
    # RISK_GATES=True: score_stocks 内部已做风控闸门, 剔除 trend_up=False(000001) -> 3 只
    assert len(s) == 3
    assert "000001" not in s["code"].tolist()
    # 降序
    vals = s["composite_score"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_score_stocks_empty():
    assert len(fe.score_stocks(pd.DataFrame())) == 0


def test_apply_risk_gates_drops_downtrend():
    df = _df()
    out = fe.apply_risk_gates(df)
    # 平安(trend_up=False) 被剔除
    assert "000001" not in out["code"].tolist()
    # 其余保留
    assert len(out) == 3


def test_apply_risk_gates_rs_falling_knife_filtered():
    df = _df()
    # 宁德 rs20=-8 边界保留; 设为 -10 应被剔除(自由落体)
    df.loc[df["code"] == "300750", "rs20"] = -10.0
    out = fe.apply_risk_gates(df)
    assert "300750" not in out["code"].tolist()


def test_pick_top_by_sector_respects_max():
    s = fe.score_stocks(_df())
    picks = fe.pick_top_by_sector(s, max_per_sector=1)
    # 银行板块 2 只, 只取 1 只
    bank = [p for p in picks if p["sector"] == "银行"]
    assert len(bank) == 1


def test_pick_top_by_sector_empty():
    assert fe.pick_top_by_sector(pd.DataFrame()) == []


def test_filter_candidates_excludes_st_and_small_cap():
    df = _df()
    df.loc[0, "name"] = "ST茅台"
    df.loc[1, "market_cap"] = 1.0e7  # 低于 min_market_cap
    out = fe.filter_candidates(df)
    assert "ST茅台" not in out["name"].tolist()
    assert 1.0e7 not in out["market_cap"].tolist()


def test_zscore_group_neutral():
    df = pd.DataFrame({
        "sector": ["A", "A", "B", "B"],
        "x": [10.0, 20.0, 100.0, 200.0],
    })
    z = fe._zscore(df, "x")
    # 行业内中性化: A 组均值 15 -> 正负标准分; B 组均值 150
    assert z.iloc[0] < 0
    assert z.iloc[1] > 0
    assert z.iloc[2] < 0
    assert z.iloc[3] > 0
