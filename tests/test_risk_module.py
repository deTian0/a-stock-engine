"""risk_module 单元测试：allocate_basket（纯函数）与 enrich_risk_metrics（mock 子进程）。

覆盖: 正常路径 / 边界（空、零/负预算、等权、全零评分）/ 异常分支（max_single 截断再分配）。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk_module import allocate_basket, enrich_risk_metrics
from conftest import make_kline_stdout, fake_subprocess_run_factory


# ---------------- allocate_basket ----------------
def test_allocate_score_weighted_normal():
    alloc = allocate_basket([80, 60, 40], 0.30, method="score_weighted")
    assert len(alloc) == 3
    assert abs(sum(alloc) - 0.30) < 1e-9
    assert alloc[0] > alloc[1] > alloc[2]  # 高分拿更多


def test_allocate_empty_scores():
    assert allocate_basket([], 0.30) == []


def test_allocate_zero_budget():
    assert allocate_basket([80, 60], 0.0) == [0.0, 0.0]


def test_allocate_negative_budget():
    assert allocate_basket([80, 60], -0.1) == [0.0, 0.0]


def test_allocate_equal_method():
    alloc = allocate_basket([80, 60, 40], 0.30, method="equal")
    assert abs(sum(alloc) - 0.30) < 1e-9
    assert alloc == pytest.approx([0.10, 0.10, 0.10], rel=1e-6)


def test_allocate_unknown_method_falls_to_equal():
    alloc = allocate_basket([80, 60], 0.30, method="bogus")
    assert alloc == [0.15, 0.15]


def test_allocate_all_zero_scores_falls_to_equal():
    alloc = allocate_basket([0, 0, 0], 0.30, method="score_weighted")
    assert abs(sum(alloc) - 0.30) < 1e-9
    assert alloc[0] == alloc[1] == alloc[2]


def test_allocate_max_single_caps_and_redistributes():
    alloc = allocate_basket([100, 1, 1], 0.30, method="score_weighted", max_single=0.08)
    # 原超限项被硬性截断到 max_single
    assert alloc[0] <= 0.08 + 1e-9
    assert abs(sum(alloc) - 0.30) < 1e-9
    # 余量补给未超限项(它们可能超过 max_single —— 函数仅对原超限项硬性截断)
    assert alloc[1] > 0.08


def test_allocate_max_single_none_no_cap():
    alloc = allocate_basket([100, 1], 0.30, method="score_weighted")
    assert alloc[0] > 0.08  # 未设上限则不受限
    assert abs(sum(alloc) - 0.30) < 1e-9


# ---------------- enrich_risk_metrics ----------------
def test_enrich_risk_metrics_empty_df():
    df = pd.DataFrame()
    out = enrich_risk_metrics(df)
    assert len(out) == 0


def test_enrich_risk_metrics_adds_columns(monkeypatch):
    # 用假子进程输出（>=15 根K线），验证 ATR 止损/流动性/集中度标签计算
    out = make_kline_stdout("sh600519", [10.0 - i * 0.1 for i in range(20)])
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run_factory(out))

    df = pd.DataFrame({
        "code": ["600519", "000001"],
        "composite_score": [80, 60],
        "sector": ["白酒", "白酒"],
        "amount": [6e8, 1e8],
        "close": [10.0, 15.0],
    })
    res = enrich_risk_metrics(df, regime_cap=0.5, total_capital=100000)
    assert "stop_loss" in res.columns
    assert "atr14" in res.columns
    assert "liquidity_tag" in res.columns
    assert "sector_warning" in res.columns
    # 600519: closes 递减 0.1/步 -> tr=0.1, atr=0.1, stop=close-2*atr=10-0.2=9.8
    row0 = res[res["code"] == "600519"].iloc[0]
    assert abs(row0["stop_loss"] - 9.8) < 1e-6
    # 同板块 2 只(<3)不应触发警告
    assert "白酒" not in (row0["sector_warning"] or "")


def test_enrich_risk_metrics_sector_concentration(monkeypatch):
    out = make_kline_stdout("sh600001", [10.0 - i * 0.1 for i in range(20)])
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run_factory(out))
    df = pd.DataFrame({
        "code": ["600001", "600002", "600003"],
        "composite_score": [70, 70, 70],
        "sector": ["半导体", "半导体", "半导体"],
        "amount": [6e8, 6e8, 6e8],
        "close": [10.0, 11.0, 12.0],
    })
    res = enrich_risk_metrics(df, regime_cap=0.5, total_capital=100000)
    # 同板块 3 只 -> 触发集中度警告(文本为"同板块N只", 不含板块名)
    assert any("同板块3只" in (w or "") for w in res["sector_warning"])
