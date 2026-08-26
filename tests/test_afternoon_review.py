"""afternoon_review 单元测试：纯助手函数 + generate_review_html（正常/边界）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import afternoon_review as ar


def test_compute_stop_loss_normal():
    closes = [10.0, 10.5, 9.8, 11.0, 10.2, 10.8]  # 波动序列, 今日 10.8
    stop = ar._compute_stop_loss(closes, 10.8)
    assert stop is not None
    assert 0 < stop < 10.8


def test_compute_stop_loss_too_short_returns_none():
    assert ar._compute_stop_loss([10.0], 10.0) is None
    assert ar._compute_stop_loss([], 10.0) is None
    assert ar._compute_stop_loss([10.0, 11.0], 0) is None


def test_fmt_chg_none():
    assert ar._fmt_chg(None) == "-"


def test_fmt_chg_signs():
    assert ar._fmt_chg(3.2) == "+3.2%"
    assert ar._fmt_chg(-1.5) == "-1.5%"


def test_esc_handles_none():
    # 注意: afternoon_review._esc 对 None 直接 str(None)="None", 不一致于 report_renderer
    assert ar._esc(None) == "None"


def test_analyze_cause_volume_and_breakout():
    # prices 按项目约定为「最新在前(降序)」: close[0]=最新价
    # 至少 20 个收盘价: close[0]=20 为近期最高 -> 触发「突破近期高点」
    prices = {"600519": [float(x) for x in range(20, 0, -1)]}
    row = {"volume_ratio": 2.0, "amplitude": 6.0, "sector": "白酒"}
    causes = ar._analyze_cause(row, prices, "600519")
    assert any("放量" in c for c in causes)
    # 最新价 20 == 近期最大 20 -> 突破近期高点
    assert any("突破近期高点" in c for c in causes)


def test_analyze_cause_no_signal_falls_back():
    prices = {"600519": [15.0, 15.0, 15.0]}
    row = {"volume_ratio": 1.0, "amplitude": 1.0, "sector": "白酒"}
    causes = ar._analyze_cause(row, prices, "600519")
    assert len(causes) >= 1


def test_generate_review_html_minimal():
    data = {
        "date": "2026-08-26",
        "generated_at": "15:30",
        "top_sectors": [{"sector": "光伏", "avg_chg": 2.3, "max_chg": 3.0,
                         "stock_count": 5, "amount_yi": 8}],
        "sector_stocks": {},
        "t0_verify": None,
        "hit_stats": {"total_hits": 1, "total_stocks": 2, "total_cycles": 1,
                      "active_count": 1, "pre_top5": [], "post_top5": []},
        "tracking_md": "",
    }
    html = ar.generate_review_html(data)
    assert "<!DOCTYPE html>" in html
    assert "盘后复盘报告" in html
    assert "光伏" in html
