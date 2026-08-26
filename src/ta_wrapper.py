"""
ta_wrapper.py — 技术指标计算（numpy 单一实现）

统一使用 numpy 实现 RSI / MACD / ATR / MA / 布林带。

历史背景：本模块曾是「ta 库加速 + numpy 回退」双实现。但 ta 既未在
requirements.txt 声明、生产/测试环境也从未安装，导致 ta 分支成为永不执行的
死代码，且 numpy 回退路径曾含 RSI 纯涨边界 bug（已修）。双实现还会漂移、
徒增测试负担。现收敛为**单一 numpy 实现**：确定性、可测试、零外部依赖。
"""

import numpy as np
import pandas as pd


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 相对强弱指标（SMA 回退，Wilder 风格）。

    返回与输入等长的数组；前 ``period-1`` 个点为预热期，返回 ``NaN``
    （与 pandas ``rolling(period)`` 惯例一致，避免用部分窗口均值给出虚假数值）。

    边界处理：
      - 纯上涨 (avg_loss == 0) -> 100
      - 纯下跌 (avg_gain == 0) -> 0
      - 横盘 (两者皆 0)        -> 50（中性）
    旧实现用 ``np.divide(where=avg_loss!=0)`` 在纯上涨时给 rs=0 -> 返回 0，
    与「RSI=100」定义相悖。
    """
    close = np.asarray(close, dtype=float)
    if len(close) == 0:
        return close
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(period) / period, mode="full")[:len(close)]
    avg_loss = np.convolve(losses, np.ones(period) / period, mode="full")[:len(close)]
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.where(avg_loss == 0,
                   np.where(avg_gain == 0, 50.0, 100.0),
                   np.where(avg_gain == 0, 0.0, rsi))
    # 预热期：部分窗口均值无意义，置 NaN
    if len(rsi) >= period:
        rsi[:period - 1] = np.nan
    else:
        rsi[:] = np.nan
    return rsi


def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: 返回 (dif, dea, macd_bar)。"""
    close = np.asarray(close, dtype=float)

    def ema(data, p):
        alpha = 2 / (p + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    bar = 2 * (dif - dea)
    return dif, dea, bar


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 真实波幅。"""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.convolve(tr, np.ones(period) / period, mode="full")[:len(close)]


def compute_ma(close: np.ndarray, period: int) -> np.ndarray:
    """简单移动均线。"""
    return np.convolve(close, np.ones(period) / period, mode="full")[:len(close)]


def compute_bb(close: np.ndarray, period: int = 20, nbdev: int = 2):
    """布林带: (upper, middle, lower)。"""
    mid = compute_ma(close, period)
    std = np.array([np.std(close[max(0, i - period + 1):i + 1]) for i in range(len(close))])
    std[:period - 1] = np.nan
    return mid + nbdev * std, mid, mid - nbdev * std
