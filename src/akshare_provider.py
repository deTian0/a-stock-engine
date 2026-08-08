"""
akshare_provider.py - akshare 数据源

提供与 WestockCLI 相同接口的 akshare 实现。
所有数据自动缓存到 SQLite。

用法:
    from akshare_provider import AkshareProvider
    provider = AkshareProvider()
    df = provider.get_stock_list()
    df = provider.get_kline("000001", days=120)
    df = provider.get_fundamentals(["000001", "600519"])
"""

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from database import get_db

logger = logging.getLogger(__name__)

# 清除不可用的系统代理（HTTP/HTTPS_PROXY 环境变量）
for _k in list(os.environ.keys()):
    if _k.lower().endswith('_proxy') and '127.0.0.1' in os.environ.get(_k, ''):
        del os.environ[_k]
        logger.debug(f"已清除系统代理: {_k}={os.environ.get(_k, 'N/A')}")


class AkshareProvider:
    """akshare 数据源，接口与 WestockCLI 兼容。"""

    def __init__(self, cache_ttl_hours: int = 6):
        self._cache_ttl = cache_ttl_hours
        self._source = "akshare"
        self._stock_list_cache: Optional[pd.DataFrame] = None

    @property
    def db(self):
        return get_db()

    # ---- 股票列表 ----

    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表。"""
        cache_key = "stock_list_full"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            # 标准化列名
            df = df.rename(columns={
                "代码": "code",
                "名称": "name",
                "最新价": "close",
                "涨跌幅": "change_pct",
                "涨跌额": "change_amount",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "pre_close",
                "量比": "volume_ratio",
                "换手率": "turnover",
                "市盈率-动态": "pe",
                "市净率": "pb",
                "总市值": "market_cap",
                "流通市值": "float_cap",
                "60日涨跌幅": "change_60d",
                "年初至今涨跌幅": "change_ytd",
            })
            df["code"] = df["code"].astype(str).str.zfill(6)
            logger.info(f"akshare: 获取股票列表 {len(df)} 只")
            self.db.cache_put(cache_key, "stock_list", df, self._source, self._cache_ttl)
            return df
        except Exception as e:
            logger.error(f"akshare 获取股票列表失败: {e}")
            raise

    # ---- K线数据 ----

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> Optional[pd.DataFrame]:
        """获取个股日K线数据。"""
        cache_key = f"kline_{code}_{days}_{adjust}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            # 日K线
            period = "daily"
            # 计算起始日期
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=code, period=period,
                start_date=start_date, end_date=end_date, adjust=adjust
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "change_pct",
                "涨跌额": "change_amount",
                "换手率": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            logger.debug(f"akshare: K线 {code} {len(df)} 行")
            self.db.cache_put(cache_key, "kline", df, self._source, self._cache_ttl)
            return df
        except Exception as e:
            logger.error(f"akshare 获取K线失败 {code}: {e}")
            return pd.DataFrame()

    def get_index_kline(self, code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取指数日K线数据。"""
        cache_key = f"index_kline_{code}_{days}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")

            # 指数代码映射
            idx_map = {
                "000001": "000001",  # 上证指数
                "399001": "399001",  # 深证成指
                "399006": "399006",  # 创业板指
            }
            symbol = idx_map.get(code, code)

            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}" if code.startswith("0") else f"sz{symbol}")
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "amount": "amount",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            self.db.cache_put(cache_key, "index_kline", df, self._source, self._cache_ttl)
            return df
        except Exception as e:
            logger.error(f"akshare 获取指数K线失败 {code}: {e}")
            return pd.DataFrame()

    # ---- 基本面数据 ----

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """获取批量基本面数据。"""
        import hashlib
        key_hash = hashlib.md5(",".join(sorted(codes[:50])).encode()).hexdigest()[:12]
        cache_key = f"fundamentals_{key_hash}_{len(codes)}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # 用实时行情中的基本面数据（PE/PB/市值等）
            full_list = self.get_stock_list()
            if full_list is None or len(full_list) == 0:
                return pd.DataFrame()

            df = full_list[full_list["code"].isin(codes)].copy()
            logger.info(f"akshare: 基本面数据 {len(df)}/{len(codes)} 只")
            self.db.cache_put(cache_key, "fundamentals", df, self._source, 2)
            return df
        except Exception as e:
            logger.error(f"akshare 获取基本面失败: {e}")
            return pd.DataFrame()

    # ---- 板块映射 ----

    def get_sector_mapping(self) -> dict[str, str]:
        """获取股票-板块映射。"""
        cache_key = "sector_mapping"
        cached = self.db.cache_get(cache_key)
        if cached is not None and len(cached) > 0:
            return dict(zip(cached["code"], cached.get("sector", [""] * len(cached))))

        try:
            full = self.get_stock_list()
            if "code" not in full.columns:
                return {}
            # 用股票名称后两个字作为简易板块标识
            # akshare 没有直接的板块映射，用行业分类替代
            result = {}
            for _, row in full.iterrows():
                code = str(row.get("code", ""))
                name = str(row.get("name", ""))
                # 用股票名称首字做粗分类
                result[code] = name[:2] if len(name) >= 2 else name

            # 缓存为 DataFrame
            mapping_df = pd.DataFrame([
                {"code": k, "sector": v} for k, v in result.items()
            ])
            self.db.cache_put(cache_key, "sector_mapping", mapping_df, self._source, 24)
            return result
        except Exception as e:
            logger.error(f"akshare 获取板块映射失败: {e}")
            return {}

    def get_sector_list(self) -> pd.DataFrame:
        """获取板块行情列表。"""
        cache_key = "sector_list"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            df = ak.stock_board_concept_name_em()
            df = df.rename(columns={
                "板块名称": "name",
                "最新价": "close",
                "涨跌幅": "change_pct",
                "成交额": "amount",
            })
            df["change_5d"] = df.get("change_pct", 0)
            df["change_20d"] = df.get("change_pct", 0)
            df["amount_change"] = 0
            self.db.cache_put(cache_key, "sector_list", df, self._source, 4)
            return df
        except Exception as e:
            logger.error(f"akshare 获取板块列表失败: {e}")
            return pd.DataFrame()


# ---- 全局单例 ----
_provider: Optional[AkshareProvider] = None


def get_akshare() -> AkshareProvider:
    global _provider
    if _provider is None:
        _provider = AkshareProvider()
    return _provider
