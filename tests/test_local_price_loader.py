"""local_price_loader 单元测试：纯计算方法（正常/边界）。

get_price/get_batch_prices 依赖 IO（KlineCache + CLI），此处仅测不依赖网络的
calc_* 方法 + get_batch_prices 在空输入下的安全行为。

注意: calc_* 是**实例方法**（self 仅占位），必须通过实例调用；用 __new__ 绕过
__init__ 的 IO（KlineCache + get_cli），避免单元测试触发网络/磁盘副作用。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from local_price_loader import LocalPriceLoader


def _loader():
    """绕过 __init__ 的 IO（KlineCache + get_cli），仅测纯计算方法。"""
    return LocalPriceLoader.__new__(LocalPriceLoader)


def _df(closes, turnover=None):
    n = len(closes)
    d = {"date": pd.date_range("2026-01-01", periods=n), "close": closes}
    if turnover is not None:
        d["turnover"] = turnover
    return pd.DataFrame(d)


def test_calc_returns_normal():
    df = _df([10.0, 11, 12, 13, 14])
    r = _loader().calc_returns(df, [1, 2])
    # calc_returns 对结果 round(,2); 期望值用同样四舍五入与源码对齐
    assert r["return_1d"] == pytest.approx(round((14/13 - 1) * 100, 2), abs=1e-9)
    assert r["return_2d"] == pytest.approx(round((14/12 - 1) * 100, 2), abs=1e-9)


def test_calc_returns_insufficient_returns_empty():
    df = _df([10.0, 11])  # 仅 1 个周期差, 最大 period=60 需要 61 行
    assert _loader().calc_returns(df, [1, 5, 20, 60]) == {}


def test_calc_volatility_normal():
    rng = np.random.default_rng(2)
    closes = 100 + np.cumsum(rng.normal(0, 1, 40))
    df = _df(closes)
    vol = _loader().calc_volatility(df, 20)
    assert np.isfinite(vol)
    assert vol > 0


def test_calc_volatility_short_returns_nan():
    df = _df([10.0, 11, 12])
    assert np.isnan(_loader().calc_volatility(df, 20))


def test_calc_momentum_normal():
    df = _df([10.0, 10, 10, 12])  # 3 个周期差, 窗口 3 -> (12/10-1)*100=20
    assert _loader().calc_momentum(df, 3) == pytest.approx(20.0, abs=1e-6)


def test_calc_momentum_short_returns_nan():
    df = _df([10.0, 11])
    assert np.isnan(_loader().calc_momentum(df, 3))


def test_calc_ma_normal():
    df = _df([10.0, 12, 14, 16, 18])
    assert _loader().calc_ma(df, 3) == pytest.approx(16.0, abs=1e-6)


def test_calc_ma_short_returns_nan():
    df = _df([10.0, 12])
    assert np.isnan(_loader().calc_ma(df, 3))


def test_calc_turnover_present():
    df = _df([10.0, 11, 12, 13, 14, 15],
             turnover=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert _loader().calc_turnover(df, 5) == pytest.approx(4.0, abs=1e-6)


def test_calc_turnover_missing_column_returns_nan():
    df = _df([10.0, 11, 12, 13, 14, 15])
    assert np.isnan(_loader().calc_turnover(df, 5))


def test_get_batch_prices_empty_input():
    loader = LocalPriceLoader.__new__(LocalPriceLoader)  # 不触发 __init__ 的 IO
    assert loader.get_batch_prices([]) == {}
