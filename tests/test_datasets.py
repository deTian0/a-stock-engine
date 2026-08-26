"""三套不同数据校验（鲁棒性）：多头完整 / 空头空候选 / 脏数据缺列。

目的: 验证 generate_brief 在数据分布与字段完整性变化下行为稳定、不崩溃、输出合理。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_brief import generate_brief


# ---- 数据集 1: 多头 + 完整候选 ----
def test_dataset_bull_full(config):
    res = {
        "timestamp": "2026-08-25 23:00", "elapsed_seconds": 3.0,
        "regime": {"regime": "多头", "position_cap": 0.8, "judgment": "强势",
                   "indices": {"000001": {"name": "上证", "close": 3500, "ma_short": 3480,
                                          "ma_long": 3400, "above_ma": True}}},
        "categories": {
            "②A_质量榜": pd.DataFrame({"code": ["600519"], "name": ["茅台"],
                                      "composite_score": [88], "roe": [28], "sector": ["白酒"]}),
            "②B_短线榜": pd.DataFrame({"code": ["000001"], "name": ["平安"],
                                      "composite_score": [72], "roe": [12]}),
        },
        "etf_picks": pd.DataFrame({"code": ["515790"], "name": ["光伏ETF"], "amount": [2e9]}),
        "l4_results": pd.DataFrame({"code": ["600519"], "name": ["茅台"], "close": [1800]}),
        "rebound_picks": [], "l2_filtered_count": 80,
        "holding_prices": {"159852": 0.663, "159869": 1.079, "159552": 2.215},
    }
    out = generate_brief(res, config)
    assert "多头" in out and "茅台" in out and "总持仓" in out


# ---- 数据集 2: 空头 + 空候选（极端防御场景） ----
def test_dataset_bear_empty(config):
    res = {
        "timestamp": "2026-08-25 15:30", "elapsed_seconds": 0.5,
        "regime": {"regime": "空头", "position_cap": 0.2, "judgment": "弱势", "indices": {}},
        "categories": {},
        "etf_picks": pd.DataFrame(),
        "l4_results": pd.DataFrame(),
        "rebound_picks": [], "l2_filtered_count": 0,
        "holding_prices": {},
    }
    out = generate_brief(res, config)
    assert "空头" in out
    assert "减仓防御" in out
    # 即便空候选，持仓追踪仍依 config 渲染（验证不崩溃）
    assert "总持仓" in out


# ---- 数据集 3: 脏数据（缺列 / 缺 judgment / 部分实时价） ----
def test_dataset_dirty_missing_columns(config):
    res = {
        "timestamp": "2026-08-25 23:00", "elapsed_seconds": 1.0,
        # regime 故意缺 'judgment'
        "regime": {"regime": "震荡", "position_cap": 0.5, "indices": {}},
        "categories": {
            # ②A 缺 roe / sector / close —— 测试 .get 兜底（composite_score 引擎恒有，保留）
            "②A_质量榜": pd.DataFrame({"code": ["600519"], "name": ["茅台"], "composite_score": [88]}),
            # ②B 缺 momentum_20d / entry_ok
            "②B_短线榜": pd.DataFrame({"code": ["000001"], "name": ["平安"], "composite_score": [72]}),
        },
        "etf_picks": pd.DataFrame({"code": ["515790"], "name": ["光伏ETF"]}),  # 缺 amount
        "l4_results": pd.DataFrame({"code": ["600519"], "name": ["茅台"]}),
        "rebound_picks": [], "l2_filtered_count": 0,
        "holding_prices": {"159852": 0.663},  # 仅部分持仓有实时价
    }
    out = generate_brief(res, config)
    assert isinstance(out, str)
    assert len(out) > 0
    # 部分实时价路径: 有价的取 live，无价的回退 cost，整体不崩溃
    assert "0.663" in out and "总持仓" in out
