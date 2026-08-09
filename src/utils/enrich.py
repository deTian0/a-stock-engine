"""
enrich_short.py - 短线因子增强模块

为 L4 结果增加短线维度的增强分析：
- 量价关系（量比、换手率突变）
- 短期趋势（5日/10日均线斜率）
- 资金流向（主力净流入）
- 技术指标（RSI、KDJ）

输出增强后的 DataFrame，供 daily_brief.py 使用。
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.local import LocalPriceLoader

logger = logging.getLogger(__name__)


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    """计算 RSI 指标。"""
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.tail(period).mean()
    avg_loss = loss.tail(period).mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calc_kdj(df: pd.DataFrame, n: int = 9) -> dict:
    """计算 KDJ 指标。"""
    if len(df) < n:
        return {"k": float("nan"), "d": float("nan"), "j": float("nan")}

    low_min = df["low"].rolling(n, min_periods=1).min()
    high_max = df["high"].rolling(n, min_periods=1).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100

    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    return {
        "k": round(float(k.iloc[-1]), 2),
        "d": round(float(d.iloc[-1]), 2),
        "j": round(float(j.iloc[-1]), 2),
    }


def calc_ma_slope(close: pd.Series, window: int) -> float:
    """计算均线斜率（归一化）。"""
    if len(close) < window * 2:
        return float("nan")
    ma = close.rolling(window).mean()
    recent = ma.tail(5)
    if recent.isna().any() or len(recent) < 2:
        return float("nan")
    slope = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100
    return round(float(slope), 2)


def calc_volume_ratio(df: pd.DataFrame, window: int = 5) -> float:
    """计算量比（当日成交量 / 近N日平均成交量）。"""
    if len(df) < window + 1:
        return float("nan")
    today_vol = df["volume"].iloc[-1]
    avg_vol = df["volume"].iloc[-(window + 1):-1].mean()
    if avg_vol > 0:
        return round(float(today_vol / avg_vol), 2)
    return float("nan")


def enrich(l4_results: pd.DataFrame, price_loader: LocalPriceLoader = None) -> pd.DataFrame:
    """
    对 L4 结果进行短线因子增强。

    Args:
        l4_results: L4 评分结果 DataFrame
        price_loader: 价格数据加载器（可选，默认新建）

    Returns:
        增强后的 DataFrame，新增列: rsi, kdj_k, kdj_d, kdj_j,
        ma5_slope, ma10_slope, volume_ratio
    """
    if price_loader is None:
        price_loader = LocalPriceLoader()

    if len(l4_results) == 0:
        return l4_results

    df = l4_results.copy()
    codes = df["code"].tolist() if "code" in df.columns else []

    for col in ["rsi", "kdj_k", "kdj_d", "kdj_j", "ma5_slope", "ma10_slope", "volume_ratio"]:
        df[col] = np.nan

    for i, code in enumerate(codes):
        try:
            price_df = price_loader.get_price(code, days=30)
            if len(price_df) < 15:
                continue

            close = price_df["close"]
            df.loc[i, "rsi"] = calc_rsi(close)
            kdj = calc_kdj(price_df)
            df.loc[i, "kdj_k"] = kdj["k"]
            df.loc[i, "kdj_d"] = kdj["d"]
            df.loc[i, "kdj_j"] = kdj["j"]
            df.loc[i, "ma5_slope"] = calc_ma_slope(close, 5)
            df.loc[i, "ma10_slope"] = calc_ma_slope(close, 10)
            df.loc[i, "volume_ratio"] = calc_volume_ratio(price_df)

        except Exception as e:
            logger.debug(f"短线增强失败 {code}: {e}")

    # 短线综合信号
    df["short_signal"] = "中性"
    df.loc[(df["rsi"] < 30) & (df["kdj_j"] < 0), "short_signal"] = "超卖"
    df.loc[(df["rsi"] > 70) & (df["kdj_j"] > 100), "short_signal"] = "超买"
    df.loc[(df["ma5_slope"] > 1) & (df["volume_ratio"] > 1.5), "short_signal"] = "放量上攻"
    df.loc[(df["ma5_slope"] < -1) & (df["volume_ratio"] > 1.5), "short_signal"] = "放量下跌"

    logger.info(f"短线增强完成: {len(df)} 只股票, 信号分布: {df['short_signal'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # 测试用
    from engine.selection import MultiFactorEngine
    engine = MultiFactorEngine()
    results = engine.run()
    enriched = enrich(results["l4_results"])
    print(enriched[["code", "composite_score", "rsi", "short_signal"]].to_string())
