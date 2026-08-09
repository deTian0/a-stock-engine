"""
local_price_loader.py - 本地价格数据加载器

从本地缓存加载股票价格数据，提供统一的接口。
当缓存不可用时自动回退到 westock-data CLI 获取。
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from engine.cache import KlineCache
from data.westock import get_cli

logger = logging.getLogger(__name__)


class LocalPriceLoader:
    """统一的价格数据加载器：优先读本地缓存，缓存失效则从CLI获取。"""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache = KlineCache(cache_dir=cache_dir + "/kline")
        self.cli = get_cli()

    def get_price(self, code: str, days: int = 120,
                  adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票价格数据（优先本地缓存）。
        返回 DataFrame，至少包含: date, open, close, high, low, volume, amount
        """
        # 先查缓存
        df = self.cache.get(code, max_age_hours=12)
        if df is not None and len(df) >= days:
            return df.tail(days).reset_index(drop=True)

        # 缓存未命中，从CLI获取
        logger.info(f"从CLI获取K线数据: {code}, {days}天")
        try:
            df = self.cli.get_kline(code, days=days, adjust=adjust)
            if df is not None and len(df) > 0:
                self.cache.put(code, df)
                return df
        except Exception as e:
            logger.error(f"获取K线数据失败 {code}: {e}")

        # 如果缓存有旧数据，返回旧数据
        if df is not None and len(df) > 0:
            logger.warning(f"使用过期缓存数据: {code}")
            return df.tail(days).reset_index(drop=True)

        return pd.DataFrame()

    def get_batch_prices(self, codes: list[str], days: int = 120) -> dict[str, pd.DataFrame]:
        """批量获取多只股票的价格数据。"""
        results = {}
        for code in codes:
            df = self.get_price(code, days=days)
            if len(df) > 0:
                results[code] = df
        return results

    def calc_returns(self, df: pd.DataFrame, periods: list[int] = None) -> dict[str, float]:
        """
        计算收益率指标。
        periods: [1, 5, 20, 60] 对应 日/周/月/季 收益率
        """
        if periods is None:
            periods = [1, 5, 20, 60]

        if len(df) < max(periods) + 1:
            return {}

        close = df["close"].values
        results = {}
        for p in periods:
            if len(close) > p:
                ret = (close[-1] / close[-1 - p] - 1) * 100
                results[f"return_{p}d"] = round(ret, 2)
        return results

    def calc_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """计算波动率（日收益率标准差 × sqrt(250)）。"""
        if len(df) < window + 1:
            return float("nan")
        returns = df["close"].pct_change().dropna().tail(window)
        return round(returns.std() * np.sqrt(250) * 100, 2)

    def calc_momentum(self, df: pd.DataFrame, window: int = 20) -> float:
        """计算动量指标（过去N日涨跌幅 %）。"""
        if len(df) < window + 1:
            return float("nan")
        close = df["close"].values
        return round((close[-1] / close[-1 - window] - 1) * 100, 2)

    def calc_ma(self, df: pd.DataFrame, window: int = 20) -> float:
        """计算移动平均线。"""
        if len(df) < window:
            return float("nan")
        return round(df["close"].tail(window).mean(), 2)

    def calc_turnover(self, df: pd.DataFrame, window: int = 5) -> float:
        """计算近N日平均换手率（如有turnover列）。"""
        if "turnover" not in df.columns or len(df) < window:
            return float("nan")
        return round(df["turnover"].tail(window).mean(), 2)
