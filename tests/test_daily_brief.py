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


def test_codes_block_copyable_per_section(base_results, config):
    """新增功能: 每个板块下增加一行可直复制进同花顺的代码, 每行最多 5 个。"""
    import re
    df = pd.DataFrame({
        "code": ["600519", "000001", "300750", "000858", "600036", "601318"],
        "name": ["茅台", "平安", "宁德", "五粮液", "招行", "平安保险"],
        "composite_score": [85, 70, 65, 80, 75, 72],
        "roe": [25, 12, 15, 20, 16, 14],
        "momentum_20d": [3, 2, 1, 4, 2, 1],
        "entry_ok": [True, True, True, True, True, True],
        "sector": ["白酒", "银行", "电池", "白酒", "银行", "保险"],
        "close": [1800, 15, 200, 150, 40, 55],
    })
    res = dict(base_results)
    res["categories"] = {"②A_质量榜": df}
    out = generate_brief(res, config)
    # 出现同花顺复制块标记
    assert "同花顺自选(复制)" in out
    # 每行复制块内 6 位代码数 <= 5
    block_lines = [ln for ln in out.splitlines() if "同花顺自选(复制)" in ln]
    assert block_lines, "未生成复制块"
    for ln in block_lines:
        codes = re.findall(r"\d{6}", ln)
        assert len(codes) <= 5, f"超过5个/行: {ln}"
    # 6 只代码全部出现在复制块中(跨行拆分)
    all_codes = [c for ln in block_lines for c in re.findall(r"\d{6}", ln)]
    for c in ["600519", "000001", "300750", "000858", "600036", "601318"]:
        assert c in all_codes


def test_tech_columns_use_enricher_fields(base_results, config):
    """修复验证: enricher 实际产出 tech_ma/tech_macd/tech_signal,
    旧列名 ma_status/macd_signal 不再生成 -> 技术面列必须显示真实值而非 '-'。"""
    df = pd.DataFrame({
        "code": ["600519"], "name": ["茅台"], "composite_score": [85],
        "roe": [25], "momentum_20d": [3], "entry_ok": [True],
        "sector": ["白酒"], "close": [1800],
        # 仅有 enricher 实际产出的列, 无旧列名 ma_status/macd_signal
        "tech_ma": ["MA20>MA60"], "tech_macd": ["金叉"], "tech_signal": ["偏多"],
    })
    res = dict(base_results)
    res["categories"] = {"②A_质量榜": df}
    out = generate_brief(res, config)
    # 技术面列应呈现 enricher 的 tech_ma/tech_macd, 而非恒为 '-'
    assert "MA20>MA60" in out
    assert "金叉" in out


def test_codes_block_once_per_section_not_per_row(base_results, config):
    """回归: 复制块必须每节仅一次(表格之后), 而非每行重复。

    旧 bug 把 _cb 写进行循环, 导致表格被 blockquote 从中打断、渲染全乱。
    中长线3行 + 短线3行 -> 复制块应仅出现 2 次(每节一次), 而非 6 次(每行一次)。
    """
    q = pd.DataFrame({
        "code": ["600519", "000001", "300750"], "name": ["茅台", "平安", "宁德"],
        "composite_score": [85, 70, 65], "roe": [25, 12, 15],
        "momentum_20d": [3, 2, 1], "entry_ok": [True, True, True],
        "sector": ["白酒", "银行", "电池"], "close": [1800, 15, 200],
    })
    s = pd.DataFrame({
        "code": ["601318", "600036", "000333"], "name": ["平安2", "招行", "美的"],
        "composite_score": [80, 75, 72], "roe": [14, 16, 18],
        "momentum_20d": [2, 2, 1], "entry_ok": [True, True, True],
        "sector": ["保险", "银行", "家电"], "close": [55, 40, 70],
    })
    res = dict(base_results)
    res["categories"] = {"②A_质量榜": q, "②B_短线榜": s}
    out = generate_brief(res, config)
    # 每节仅一次: ②A(3行) + ②B(3行) + 七、持仓(base_results 含持仓) = 3 块
    # 若旧 bug 把块写进行循环, 应为 6 块(每行一次) -> 用 "<行数" 强约束
    n_blocks = out.count("同花顺自选(复制)")
    assert n_blocks == 3, f"复制块应每节一次(②A+②B+持仓=3), 实际 {n_blocks}"
    assert n_blocks < 6, "复制块被插进行循环(每行一次=6块)——回归!"


def test_header_no_double_percent(base_results, config):
    """回归: 表头仓位列应为 '仓位%', 不得残留 '仓位%%' 转义 bug。"""
    df = pd.DataFrame({
        "code": ["600519"], "name": ["茅台"], "composite_score": [85],
        "roe": [25], "momentum_20d": [3], "entry_ok": [True],
        "sector": ["白酒"], "close": [1800],
    })
    res = dict(base_results)
    res["categories"] = {"②A_质量榜": df}
    out = generate_brief(res, config)
    assert "仓位%%" not in out
    assert "|仓位%|" in out
