"""
ta_wrapper.py — 技术指标计算（ta库加速 + numpy回退）

借鉴 Qbot 使用 TA-Lib 的思路，用 ta 库替代手写的逐条计算。
ta 是纯 Python 无 C 依赖，速度够用。
"""

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 相对强弱指标。"""
    if _TA_AVAILABLE and len(close) > period:
        s = pd.Series(close)
        return ta.momentum.RSIIndicator(s, window=period).rsi().values
    # numpy fallback
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(period)/period, mode='full')[:len(close)]
    avg_loss = np.convolve(losses, np.ones(period)/period, mode='full')[:len(close)]
    # RS = avg_gain / avg_loss; 边界: 纯上涨(avg_loss==0)→RSI=100, 纯下跌(avg_gain==0)→0,
    # 横盘(两者皆0)→50(中性)。原实现用 np.divide(where=avg_loss!=0) 在纯上涨时给 rs=0
    # → 返回 0, 与「RSI=100」定义相悖 (见 test_compute_rsi_monotonic_up_is_100)。
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.where(avg_loss == 0,
                   np.where(avg_gain == 0, 50.0, 100.0),
                   np.where(avg_gain == 0, 0.0, rsi))
    return rsi


def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: 返回 (dif, dea, macd_bar)。"""
    if _TA_AVAILABLE and len(close) > slow:
        s = pd.Series(close)
        macd = ta.trend.MACD(s, window_slow=slow, window_fast=fast, window_sign=signal)
        return macd.macd_diff().values, macd.macd_signal().values, macd.macd().values
    # numpy EMA fallback
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    bar = 2 * (dif - dea)
    return dif, dea, bar


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 真实波幅。"""
    if _TA_AVAILABLE and len(close) > period:
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        return ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range().values
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.convolve(tr, np.ones(period)/period, mode='full')[:len(close)]


def compute_ma(close: np.ndarray, period: int) -> np.ndarray:
    """简单移动均线。"""
    return np.convolve(close, np.ones(period)/period, mode='full')[:len(close)]


def compute_bb(close: np.ndarray, period: int = 20, nbdev: int = 2):
    """布林带: (upper, middle, lower)。"""
    mid = compute_ma(close, period)
    std = np.array([np.std(close[max(0,i-period+1):i+1]) for i in range(len(close))])
    std[:period-1] = np.nan
    return mid + nbdev * std, mid, mid - nbdev * std
