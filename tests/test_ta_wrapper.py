"""ta_wrapper 单元测试：技术指标（正常/边界/异常）。

注意: compute_rsi/compute_macd/compute_atr 在有 ta 库时走 ta 分支、否则走 numpy 回退。
测试只断言两种实现都满足的**不变量**，保证确定性。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import ta_wrapper


def test_compute_ma_matches_pandas_rolling():
    close = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ma = ta_wrapper.compute_ma(close, 3)
    import pandas as pd
    expected = pd.Series(close).rolling(3).mean().to_numpy()
    # compute_ma 用 convolve 全窗口, 前 period-1 为部分窗口均值; 自 i>=period-1 起
    # 与 pandas rolling(period).mean() 的尾部均值完全一致
    np.testing.assert_allclose(ma[2:], expected[2:], atol=1e-9)


def test_compute_ma_length_preserved():
    close = np.linspace(10, 20, 25)
    ma = ta_wrapper.compute_ma(close, 5)
    assert len(ma) == len(close)


def test_compute_rsi_monotonic_up_is_100():
    # 严格单调递增 -> RSI 必为 100（ta 与回退两分支都如此）
    close = np.arange(1, 40, dtype=float)
    rsi = ta_wrapper.compute_rsi(close, 14)
    assert rsi[-1] == pytest.approx(100.0, abs=1e-6)


def test_compute_rsi_in_range():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 60))
    rsi = ta_wrapper.compute_rsi(close, 14)
    assert np.all((rsi >= 0) & (rsi <= 100))


def test_compute_rsi_short_series_fallback_no_crash():
    # 短序列即使 ta 不可用也应返回有限值（不抛、不 NaN 爆炸）
    close = np.array([1.0, 2.0, 3.0])
    rsi = ta_wrapper.compute_rsi(close, 14)
    assert np.all(np.isfinite(rsi))


def test_compute_macd_bar_invariant():
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, 80))
    dif, dea, bar = ta_wrapper.compute_macd(close, 12, 26, 9)
    assert len(dif) == len(close)
    np.testing.assert_allclose(bar, 2 * (dif - dea), atol=1e-6)


def test_compute_atr_positive_for_moving_series():
    n = 40
    high = np.arange(1, n + 1, dtype=float) + 1
    low = np.arange(1, n + 1, dtype=float) - 1
    close = np.arange(1, n + 1, dtype=float).astype(float)
    atr = ta_wrapper.compute_atr(high, low, close, 14)
    assert len(atr) == n
    assert np.all(atr[14:] > 0)


def test_compute_bb_ordering():
    close = np.linspace(10, 50, 60)
    upper, mid, lower = ta_wrapper.compute_bb(close, 20, 2)
    assert len(upper) == len(close)
    valid = ~np.isnan(upper)
    assert np.all(upper[valid] >= mid[valid] - 1e-6)
    assert np.all(mid[valid] >= lower[valid] - 1e-6)
