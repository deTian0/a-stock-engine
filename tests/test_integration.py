"""集成测试：串联 generate_brief 全管线 + score_lvrev→allocate_basket 篮子分配。

验证: 多 regime、含候选/ETF/持仓的完整简报渲染不崩溃且结构完整；
lvrev 内核评分后经 allocate_basket 分配，预算严格守恒。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_brief import generate_brief
from lvrev_scorer import score_lvrev
from risk_module import allocate_basket


def _build_full_results(regime, holding_prices):
    return {
        "timestamp": "2026-08-25 23:00",
        "elapsed_seconds": 2.0,
        "regime": regime,
        "categories": {
            "②A_质量榜": pd.DataFrame({
                "code": ["600519"], "name": ["茅台"], "composite_score": [85],
                "roe": [25], "momentum_20d": [3], "entry_ok": [True],
                "sector": ["白酒"], "close": [1800],
            }),
            "②B_短线榜": pd.DataFrame({
                "code": ["000001"], "name": ["平安"], "composite_score": [70],
                "roe": [12], "momentum_20d": [2], "entry_ok": [False],
                "sector": ["银行"], "close": [15],
            }),
            "③C_观察名单": pd.DataFrame({
                "code": ["300750"], "name": ["宁德"], "composite_score": [60],
                "roe": [20], "momentum_20d": [1], "entry_ok": [False],
                "sector": ["电池"], "close": [200],
            }),
        },
        "etf_picks": pd.DataFrame({
            "code": ["515790"], "name": ["光伏ETF"], "momentum": [5], "amount": [1e9],
        }),
        "l4_results": pd.DataFrame({
            "code": ["600519", "000001"], "name": ["茅台", "平安"], "close": [1800, 15],
        }),
        "rebound_picks": [],
        "l2_filtered_count": 50,
        "holding_prices": holding_prices,
    }


def test_full_brief_bull(config):
    regime = {"regime": "多头", "position_cap": 0.80, "judgment": "j",
              "indices": {"000001": {"name": "上证", "close": 3500, "ma_short": 3480,
                                     "ma_long": 3400, "above_ma": True}}}
    res = _build_full_results(regime, {"159852": 0.663, "159869": 1.079, "159552": 2.215})
    out = generate_brief(res, config)
    assert "多头" in out
    assert "茅台" in out and "平安" in out and "宁德" in out
    assert "光伏ETF" in out
    assert "0.663" in out
    assert "总持仓" in out
    assert "可适度加仓" in out  # 多头判定


def test_full_brief_bear(config):
    regime = {"regime": "空头", "position_cap": 0.20, "judgment": "j", "indices": {}}
    res = _build_full_results(regime, {"159852": 0.663, "159869": 1.079})
    out = generate_brief(res, config)
    assert "空头" in out
    assert "减仓防御" in out  # 空头判定


def test_pipeline_score_then_allocate():
    df = pd.DataFrame({
        "vol20": [0.2, 0.4, 0.6], "rev_chg": [-0.05, 0, 0.05],
        "debt_ratio": [40, 50, 60], "revenue_growth": [10, 5, 0],
    })
    scored = score_lvrev(df)
    assert "composite_score" in scored.columns
    scores = scored["composite_score"].tolist()
    alloc = allocate_basket(scores, 0.30, method="score_weighted", max_single=0.08)
    assert abs(sum(alloc) - 0.30) < 1e-9
    assert all(a >= -1e-9 for a in alloc)        # 无负值
    assert max(alloc) <= 0.30 + 1e-9             # 单只不超过总预算


def test_brief_without_holding_prices_still_renders(config):
    regime = {"regime": "震荡", "position_cap": 0.5, "judgment": "j", "indices": {}}
    res = _build_full_results(regime, None)  # 不注入实时价
    out = generate_brief(res, config)
    assert isinstance(out, str) and len(out) > 0
    assert "总持仓" in out  # 持仓追踪仍依 config 渲染
