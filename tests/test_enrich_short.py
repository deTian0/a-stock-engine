"""enrich_short 单元测试：短线因子纯计算 + enrich（mock 价格源）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import enrich_short as es


class FakeLoader:
    def get_price(self, code, days=30):
        rng = np.random.default_rng(0)
        n = 30
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=n),
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": np.full(n, 1_000_000.0),
        })


def test_calc_rsi_increasing_is_100():
    s = pd.Series(np.linspace(10, 20, 20))
    assert es.calc_rsi(s, 14) == pytest.approx(100.0, abs=1e-6)


def test_calc_kdj_returns_finite():
    df = pd.DataFrame({
        "high": np.arange(10, 40.0),
        "low": np.arange(5, 35.0),
        "close": np.arange(7, 37.0),
    })
    kdj = es.calc_kdj(df, 9)
    assert set(kdj) == {"k", "d", "j"}
    assert all(np.isfinite(v) for v in kdj.values())


def test_calc_ma_slope_positive_for_uptrend():
    s = pd.Series(np.linspace(10, 20, 30))
    slope = es.calc_ma_slope(s, 5)
    assert slope > 0


def test_calc_volume_ratio_above_one():
    df = pd.DataFrame({"volume": [1000] * 10 + [5000]})  # 今日放量
    vr = es.calc_volume_ratio(df, 5)
    assert vr > 1


def test_enrich_adds_columns(monkeypatch):
    monkeypatch.setattr(es, "LocalPriceLoader", lambda *a, **k: FakeLoader())
    df = pd.DataFrame({"code": ["600519", "000001"], "composite_score": [80, 60]})
    out = es.enrich(df)
    for col in ["rsi", "kdj_k", "kdj_d", "kdj_j", "ma5_slope", "ma10_slope", "volume_ratio", "short_signal"]:
        assert col in out.columns


def test_enrich_short_signal_classification(monkeypatch):
    monkeypatch.setattr(es, "LocalPriceLoader", lambda *a, **k: FakeLoader())
    df = pd.DataFrame({"code": ["600519"], "composite_score": [80]})
    out = es.enrich(df)
    assert out["short_signal"].iloc[0] in ("中性", "超卖", "超买", "放量上攻", "放量下跌")
