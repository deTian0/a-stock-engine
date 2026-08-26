"""daily_brief.generate_brief 单元测试：持仓修复点 + 各 section 渲染 + 鲁棒性。

重点验证 2026-08-25 修复：注入 results['holding_prices'] 时持仓"当日股价"必须取实时价，
而非回退 config 的 cost_price（旧 bug）。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_brief import generate_brief


def test_sections_present(base_results, config):
    out = generate_brief(base_results, config)
    assert isinstance(out, str)
    assert "盘前选股简报" in out
    assert "一、市场环境判断" in out
    assert "七、持仓追踪与建议" in out
    assert "总持仓" in out


def test_holdings_uses_injected_live_price(base_results, config):
    # 修复验证：注入 live 价必须覆盖 cost
    res = dict(base_results)
    res["holding_prices"] = {"159852": 0.663, "159869": 1.079, "159552": 2.215}
    out = generate_brief(res, config)
    assert "0.663" in out and "1.079" in out and "2.215" in out
    # 成本列应为真实成本（非旧 bug 的 0.725/1.142）
    assert "0.683" in out and "1.122" in out and "2.284" in out


def test_holdings_falls_back_to_cost_without_injection(base_results, config):
    # 无注入时回退到 cost（记录 fallback 行为，区别于修复后的 live 路径）
    out = generate_brief(base_results, config)
    assert "0.683" in out  # 159852 成本作为价格列出现


def test_no_holdings_config_renders_placeholder(base_results, config):
    cfg = {k: v for k, v in config.items()}
    cfg["account"] = {k: v for k, v in config.get("account", {}).items() if k != "holdings"}
    out = generate_brief(base_results, cfg)
    assert "暂无持仓配置" in out


def test_missing_indices_does_not_crash(config):
    res = {
        "timestamp": "t", "elapsed_seconds": 1.0,
        "regime": {"regime": "震荡", "position_cap": 0.5, "indices": {}},
        "categories": {}, "etf_picks": pd.DataFrame(),
        "l4_results": pd.DataFrame(), "rebound_picks": [], "l2_filtered_count": 0,
    }
    out = generate_brief(res, config)
    assert "震荡" in out


def test_with_categories_renders_candidates(base_results, config):
    df = pd.DataFrame({
        "code": ["600519"], "name": ["茅台"], "composite_score": [85],
        "roe": [25], "momentum_20d": [3], "entry_ok": [True],
        "sector": ["白酒"], "close": [1800],
    })
    res = dict(base_results)
    res["categories"] = {"②A_质量榜": df}
    out = generate_brief(res, config)
    assert "茅台" in out


def test_etf_settlement_labels(base_results, config):
    # 货币/债券/黄金/跨境类(T+0) vs 其余股票型(T+1)
    res = dict(base_results)
    res["etf_picks"] = pd.DataFrame({
        "code": ["511880", "515790"], "name": ["货币ETF", "光伏ETF"],
        "momentum": [0, 5], "amount": [1e9, 1e9],
    })
    out = generate_brief(res, config)
    # 511880 -> T+0; 515790 -> T+1
    assert "T+0" in out and "T+1" in out
