"""
akshare_provider.py - akshare 数据源（家庭网络适配版，2026-08-11 改造）

提供与 WestockCLI 相同接口的 akshare 实现。所有数据自动缓存到 SQLite。

后端选择（实测结论）：
- 家里电脑无火绒拦截，Python 出网正常；但东方财富「数据 API 子路径」
  (push2*.eastmoney.com/api/qt/...) 直连被网络层 RST 复位，Clash 本地代理
  (127.0.0.1:7897) 又间歇性抽风。故东财后端（stock_zh_a_hist / spot_em /
  board_concept_name_em）在本机不可用。
- 新浪 / 腾讯 数据接口直连可用，因此 K线走新浪、实时行情走腾讯。
  为满足 westock_cli 兜底接口，get_kline / get_index_kline 默认新浪直连。

强制直连：导入时清除所有 *_proxy 环境变量并置 NO_PROXY='*'，使 akshare 内部
requests 走直连（绕开抽风的本地 Clash），腾讯实时接口再显式 proxies=None 兜底。

用法:
    from akshare_provider import AkshareProvider, get_akshare
    provider = get_akshare()
    df = provider.get_kline("000001", days=120)          # 新浪日线
    q  = provider.get_realtime_quote("000001")            # 腾讯实时
"""

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

# ---- 强制直连：清除可能残留的本地代理（含 Clash 127.0.0.1:7897） ----
for _k in list(os.environ.keys()):
    if _k.lower().endswith("_proxy"):
        del os.environ[_k]
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from database import get_db

logger = logging.getLogger(__name__)

_DIRECT = {"http": None, "https": None}


def _prefix(code: str) -> str:
    """6/9 开头沪市用 sh，其余 sz。"""
    code = str(code).zfill(6)
    return "sh" if code.startswith(("6", "9")) else "sz"


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
        """获取全A股股票列表。东财 spot 优先，失败回退新浪代码表（仅 code/name）。"""
        cache_key = "stock_list_full"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            df = None
            try:
                df = ak.stock_zh_a_spot_em()  # 东财行情（含 PE/PB/市值）
                df = df.rename(columns={
                    "代码": "code", "名称": "name", "最新价": "close",
                    "涨跌幅": "change_pct", "涨跌额": "change_amount",
                    "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
                    "最高": "high", "最低": "low", "今开": "open",
                    "昨收": "pre_close", "量比": "volume_ratio",
                    "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb",
                    "总市值": "market_cap", "流通市值": "float_cap",
                    "60日涨跌幅": "change_60d", "年初至今涨跌幅": "change_ytd",
                })
            except Exception as e:
                logger.warning(f"akshare 东财行情列表失败，回退新浪代码表: {e}")
                df = ak.stock_info_a_code_name()  # 新浪代码表（code/name）

            if df is None or len(df) == 0:
                raise RuntimeError("akshare 股票列表为空")

            df = df.rename(columns={"代码": "code", "名称": "name"})
            df["code"] = df["code"].astype(str).str.zfill(6)
            logger.info(f"akshare: 获取股票列表 {len(df)} 只")
            self.db.cache_put(cache_key, "stock_list", df, self._source, self._cache_ttl)
            return df
        except Exception as e:
            logger.error(f"akshare 获取股票列表失败: {e}")
            raise

    # ---- K线数据（新浪直连） ----

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> Optional[pd.DataFrame]:
        """获取个股日K线（新浪 stock_zh_a_daily，直连可用）。返回 date/open/high/low/close/volume/amount。"""
        cache_key = f"kline_{code}_{days}_{adjust}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            adj = "" if adjust in ("", "bfq") else adjust  # 新浪: ''=不复权
            df = ak.stock_zh_a_daily(symbol=f"{_prefix(code)}{code}", adjust=adj)
            if df is None or len(df) == 0:
                return pd.DataFrame()

            # 中文列名归一化（新浪部分版本返回中文）
            df = df.rename(columns={
                "开盘": "open", "收盘": "close", "最高": "high",
                "最低": "low", "成交量": "volume", "成交额": "amount",
            })
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            logger.debug(f"akshare: K线 {code} {len(df)} 行（新浪）")
            self.db.cache_put(cache_key, "kline", df, self._source, self._cache_ttl)
            return df
        except Exception as e:
            logger.error(f"akshare 获取K线失败 {code}: {e}")
            return pd.DataFrame()

    def get_index_kline(self, code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取指数日K线（新浪 stock_zh_index_daily，直连可用）。"""
        cache_key = f"index_kline_{code}_{days}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=f"{_prefix(code)}{code}")
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.rename(columns={
                "开盘": "open", "收盘": "close", "最高": "high",
                "最低": "low", "成交量": "volume", "成交额": "amount",
            })
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            logger.debug(f"akshare: 指数K线 {code} {len(df)} 行（新浪）")
            self.db.cache_put(cache_key, "index_kline", df, self._source, self._cache_ttl)
            return df
        except Exception as e:
            logger.error(f"akshare 获取指数K线失败 {code}: {e}")
            return pd.DataFrame()

    # ---- 实时行情（腾讯直连） ----

    def get_realtime_quote(self, code: str) -> dict:
        """腾讯实时行情 qt.gtimg.cn（直连可用）。返回 {name, code, price, prev_close, open}。"""
        url = f"https://qt.gtimg.cn/q={_prefix(code)}{code}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=12, proxies=_DIRECT)
        # 形如 v_sz000001="1~名称~代码~当前价~昨收~今开~..."
        body = r.text.split('="', 1)[1].rstrip('";\n')
        f = body.split("~")
        return {
            "name": f[1],
            "code": f[2],
            "price": float(f[3]) if f[3] else None,
            "prev_close": float(f[4]) if f[4] else None,
            "open": float(f[5]) if f[5] else None,
        }

    # ---- 基本面数据 ----

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """获取批量基本面数据（依赖股票列表的 PE/PB/市值字段）。"""
        import hashlib
        key_hash = hashlib.md5(",".join(sorted(codes[:50])).encode()).hexdigest()[:12]
        cache_key = f"fundamentals_{key_hash}_{len(codes)}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
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
        """获取股票-板块映射（无网络，用股票名称前2字作粗分类）。"""
        cache_key = "sector_mapping"
        cached = self.db.cache_get(cache_key)
        if cached is not None and len(cached) > 0:
            return dict(zip(cached["code"], cached.get("sector", [""] * len(cached))))

        try:
            full = self.get_stock_list()
            if "code" not in full.columns:
                return {}
            result = {}
            for _, row in full.iterrows():
                code = str(row.get("code", ""))
                name = str(row.get("name", ""))
                result[code] = name[:2] if len(name) >= 2 else name

            mapping_df = pd.DataFrame([{"code": k, "sector": v} for k, v in result.items()])
            self.db.cache_put(cache_key, "sector_mapping", mapping_df, self._source, 24)
            return result
        except Exception as e:
            logger.error(f"akshare 获取板块映射失败: {e}")
            return {}

    def get_sector_list(self) -> pd.DataFrame:
        """获取板块行情列表（东财概念板，本机可能不可用，作为最后兜底）。"""
        cache_key = "sector_list"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak
            df = ak.stock_board_concept_name_em()
            df = df.rename(columns={
                "板块名称": "name", "最新价": "close",
                "涨跌幅": "change_pct", "成交额": "amount",
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


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "002223"
    print(f"=== 测试 akshare 数据拉取(直连): {code} ===")
    try:
        k = get_akshare().get_kline(code, days=5)
        print(f"[新浪日线] OK 行数={len(k)} 末3行:\n{k.tail(3).to_string(index=False)}")
    except Exception as e:
        print(f"[新浪日线] ERR {e}")
    try:
        q = get_akshare().get_realtime_quote(code)
        print(f"[腾讯实时] OK {q}")
    except Exception as e:
        print(f"[腾讯实时] ERR {e}")
    print("AKSHARE_PROVIDER_SELFTEST_DONE")
