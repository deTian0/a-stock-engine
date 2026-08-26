"""reward_attribution 单元测试：calc_attribution（纯计算，mock 价格源）。"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import reward_attribution as ra


class FakeLoader:
    def get_price(self, code, days=120):
        dates = pd.date_range("2026-01-01", periods=30)
        close = np.linspace(10.0, 13.0, 30)
        return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ra, "LocalPriceLoader", lambda *a, **k: FakeLoader())
    monkeypatch.setattr(ra, "get_cli", lambda *a, **k: object())
    monkeypatch.setattr(ra, "sector_of", lambda code, mapping=None: "白酒")
    return ra.RewardAttribution()


def test_calc_attribution_basic(patched):
    holdings = {"600519": {"name": "茅台", "shares": 100, "cost_price": 10.0}}
    start = "2026-01-01"
    end = "2026-01-30"
    res = patched.calc_attribution(holdings, start, end)
    assert res["total_return"] == pytest.approx(30.0, abs=1e-6)  # (1300/1000-1)*100
    assert res["details"][0]["hold_return"] == pytest.approx(30.0, abs=1e-6)
    assert res["details"][0]["period_return"] == pytest.approx(30.0, abs=1e-6)


def test_calc_attribution_empty(patched):
    res = patched.calc_attribution({}, "2026-01-01")
    assert res["total_return"] == 0
    assert res["details"] == []


def test_calc_attribution_loss(patched):
    holdings = {"600519": {"name": "茅台", "shares": 100, "cost_price": 20.0}}
    res = patched.calc_attribution(holdings, "2026-01-01", "2026-01-30")
    # 现价 13 < 成本 20 -> 亏损
    assert res["details"][0]["hold_return"] < 0
    assert res["total_return"] < 0


def test_generate_report_runs(patched):
    holdings = {"600519": {"name": "茅台", "shares": 100, "cost_price": 10.0}}
    res = patched.calc_attribution(holdings, "2026-01-01", "2026-01-30")
    rep = patched.generate_report(res, "2026-01-01", "2026-01-30")
    assert "收益归因分析" in rep
    assert "总收益率" in rep
