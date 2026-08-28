"""data_enricher 单元测试：纯函数 + 全字段 df 的 enrich_and_report（无网络）。

新增: enrich_amount 补全(amount/量比) + 概念板块异常重试(治 too many values to unpack)。
所有外部依赖(westock_helpers / tushare_provider / database)均通过 monkeypatch 注入内存假模块,
保证确定性、可离线运行。
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import data_enricher as de


def _fake_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


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


# ---------------------------------------------------------------------------
# enrich_amount: 成交额/量比补全（流动性标签与 ETF 成交额的前提列）
# ---------------------------------------------------------------------------
def test_enrich_amount_fills_missing(monkeypatch):
    """缺失/<=0 的 amount/量比 -> 走 westock batch_quotes 实时补全。"""
    fake_ws = _fake_module(
        "westock_helpers",
        batch_quotes=lambda codes: {
            "600519": {"amount": 1e9, "volume_ratio": 1.2},
            "000001": {"amount": 5e8, "volume_ratio": 0.9},
        },
    )
    monkeypatch.setitem(sys.modules, "westock_helpers", fake_ws)

    df = pd.DataFrame({
        "code": ["600519", "000001"],
        "close": [1800.0, 15.0],
        "amount": [np.nan, 0.0],
        "volume_ratio": [np.nan, np.nan],
    })
    out = de.enrich_amount(df)
    assert out["amount"].iloc[0] == 1e9
    assert out["amount"].iloc[1] == 5e8
    assert out["volume_ratio"].iloc[0] == 1.2
    assert out["volume_ratio"].iloc[1] == 0.9


def test_enrich_amount_preserves_valid(monkeypatch):
    """已有有效 amount/量比 -> 绝不覆盖（即使实时返回不同值）。"""
    fake_ws = _fake_module(
        "westock_helpers",
        batch_quotes=lambda codes: {"600519": {"amount": 999.0, "volume_ratio": 9.9}},
    )
    monkeypatch.setitem(sys.modules, "westock_helpers", fake_ws)

    df = pd.DataFrame({
        "code": ["600519"],
        "close": [1800.0],
        "amount": [1e9],
        "volume_ratio": [1.2],
    })
    out = de.enrich_amount(df)
    assert out["amount"].iloc[0] == 1e9
    assert out["volume_ratio"].iloc[0] == 1.2


def test_enrich_amount_empty_df():
    """空 df / 无 code 列 -> 直接返回, 不报错。"""
    assert len(de.enrich_amount(pd.DataFrame())) == 0
    assert "x" not in de.enrich_amount(pd.DataFrame({"other": [1]})).columns


# ---------------------------------------------------------------------------
# 概念板块异常重试: 治 "too many values to unpack" 被静默吞掉
# ---------------------------------------------------------------------------
def test_enrich_l4_concept_retry_on_exception(monkeypatch):
    """首次 get_concept_stats 抛异常 -> 清缓存 -> 第二次成功, 概念补全不静默失败。"""
    class FakeTS:
        def __init__(self):
            self.calls = 0

        def get_concept_stats(self):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("too many values to unpack")
            return pd.DataFrame({
                "code": ["600519", "000001"],
                "concept_name": ["白酒", "银行"],
                "concept_chg": [1.2, -0.5],
            })

    ts = FakeTS()
    monkeypatch.setitem(
        sys.modules, "tushare_provider",
        _fake_module("tushare_provider", get_tushare=lambda: ts),
    )
    # get_db 返回的 fake db 的 conn 需支持 execute/commit; _clear_concept_cache 已 try 包裹
    fake_conn = types.SimpleNamespace(
        execute=lambda *a, **k: None, commit=lambda *a, **k: None)
    monkeypatch.setitem(
        sys.modules, "database",
        _fake_module("database", get_db=lambda: types.SimpleNamespace(conn=fake_conn)),
    )

    df = pd.DataFrame({
        "code": ["600519", "000001"],
        "name": ["茅台", "平安"],
        "close": [1800.0, 15.0],
        "amount": [1e9, 5e8],          # 有效 -> enrich_amount 早退, 不触发 westock
        "volume_ratio": [1.2, 0.9],
        "composite_score": [85.0, 60.0],
        "tech_signal": ["偏多", "偏空"],  # 已有 -> 不触发 westock technical
        # 无 concept_name -> 触发概念补全分支
    })
    out = de.enrich_l4_results(df)
    # 第一次异常 + 第二次成功 = 2 次调用
    assert ts.calls == 2
    assert "concept_name" in out.columns
    assert out["concept_name"].iloc[0] == "白酒"
    assert out["concept_name"].iloc[1] == "银行"
