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


def test_review_sectors_t0_verify_handles_null_score(monkeypatch):
    """第三节早盘验证：composite_score 为 NULL（如 ETF 行）时不应崩溃。

    回归测试：修复前 `score:.0f` 对 None 抛 TypeError 被 except 吞掉，
    导致整节「早盘推荐当日表现」静默为空。
    """
    fake_picks = [
        {"code": "601965", "name": "中国汽研", "category": "②A_质量榜", "composite_score": 94.8},
        {"code": "515050", "name": "5GETF", "category": "ETF组合", "composite_score": None},
    ]

    class FakeDB:
        def get_latest_run(self):
            return {"run_id": 1, "date": "2026-08-27"}

        def get_run_detail(self, run_id):
            return {"picks": fake_picks, "factors": []}

        @property
        def conn(self):
            raise RuntimeError("conn 不应在 section 三被用到（已由 except 捕获）")

    class FakeCLI:
        def get_sector_mapping(self, codes=None):
            return {}

    monkeypatch.setattr(ar, "get_db", lambda: FakeDB())
    monkeypatch.setattr(ar, "get_cli", lambda: FakeCLI())
    # 五/六节 pick_tracker 调用打桩，避免触碰真实 DB
    monkeypatch.setattr(ar, "get_tracking_summary", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ar, "get_tracking_report", lambda *a, **k: "")

    # all_stocks 为空 -> 一/二节走「板块数据暂不可用」分支，聚焦验证第三节
    content, data = ar.review_sectors(FakeCLI(), None, pd.DataFrame(), {}, {})

    assert data.get("t0_verify") is not None
    total = sum(len(v) for v in data["t0_verify"]["by_cat"].values())
    assert total == 2, f"应汇总 2 只选股, 实际 {total}"
    # None 评分的 ETF 行应安全渲染为 '-'，而非让整节崩溃
    assert "中国汽研" in content
    assert "5GETF" in content
    assert "T+0验证失败" not in content
