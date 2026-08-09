"""
sina_provider.py - 新浪财经数据源

提供与 AkshareProvider 相同接口的新浪 API 实现。
无需 token，无需安装额外包，纯 HTTP。

用法:
    from sina_provider import SinaProvider
    provider = SinaProvider()
    df = provider.get_stock_list()
    df = provider.get_index_kline("000001", days=60)
"""

import logging
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from database import get_db

logger = logging.getLogger(__name__)

SINA_BASE = "http://hq.sinajs.cn/list="
HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}

# 指数代码 → 新浪代码
INDEX_MAP = {
    "000001": "s_sh000001",  # 上证
    "399001": "s_sz399001",  # 深证
    "399006": "s_sz399006",  # 创业板
    "000688": "s_sh000688",  # 科创50
    "000300": "s_sh000300",  # 沪深300
    "000905": "s_sh000905",  # 中证500
    "000016": "s_sh000016",  # 上证50
}

# 代码 → 新浪前缀
def _sina_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


class SinaProvider:
    """新浪财经数据源。"""

    def __init__(self, cache_ttl_hours: int = 1):
        self._cache_ttl = cache_ttl_hours
        self._source = "sina"
        self._base_codes: list[str] = []  # 从 DB 加载基础代码列表

    @property
    def db(self):
        return get_db()

    def _get_base_codes(self) -> list[str]:
        """从现有数据库获取全A股代码列表。"""
        if self._base_codes:
            return self._base_codes

        try:
            from database import get_market_db
            mkt = get_market_db()
            c = mkt.conn
            # 取最近一个交易日的所有股票代码
            latest = c.execute("SELECT MAX(date) FROM daily_price").fetchone()[0]
            rows = c.execute(
                "SELECT DISTINCT code FROM daily_price WHERE date=?",
                (latest,)
            ).fetchall()
            self._base_codes = [r[0].replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
                                for r in rows]
            mkt.close()
            logger.info(f"sina: 从DB加载 {len(self._base_codes)} 个基础代码")
        except Exception:
            self._base_codes = []

        return self._base_codes

    # ---- 股票列表（实时行情）----

    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股实时行情。"""
        base = self._get_base_codes()
        if not base:
            logger.warning("sina: 无基础代码列表，返回空")
            return pd.DataFrame()

        cache_key = "sina_stock_list"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        all_rows = []
        batch_size = 80  # 新浪单次请求上限约 80-100 只
        total = len(base)

        for i in range(0, total, batch_size):
            batch = base[i:i + batch_size]
            codes_str = ",".join(_sina_code(c) for c in batch)
            url = SINA_BASE + codes_str

            try:
                req = urllib.request.Request(url, headers=HEADERS)
                resp = urllib.request.urlopen(req, timeout=10)
                text = resp.read().decode("gbk")
            except Exception as e:
                logger.warning(f"sina batch {i//batch_size} 失败: {e}")
                continue

            for line in text.strip().split("\n"):
                try:
                    if '=""' in line or '"' not in line:
                        continue
                    name = line.split("=")[0].replace("var hq_str_", "")
                    raw = line.split('"')[1]
                    fields = raw.split(",")
                    if len(fields) < 32:
                        continue

                    # 字段映射: name,open,pre_close,close,high,low,buy,sell,vol,amount,...
                    all_rows.append({
                        "code": name[2:],  # sh600519 → 600519
                        "name": fields[0],
                        "open": float(fields[1]) if fields[1] else 0,
                        "pre_close": float(fields[2]) if fields[2] else 0,
                        "close": float(fields[3]) if fields[3] else 0,
                        "high": float(fields[4]) if fields[4] else 0,
                        "low": float(fields[5]) if fields[5] else 0,
                        "volume": float(fields[8]) if fields[8] else 0,
                        "amount": float(fields[9]) if fields[9] else 0,
                    })
                except Exception:
                    continue

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.debug(f"sina 行情: {i + batch_size}/{total}")

        df = pd.DataFrame(all_rows)
        if len(df) == 0:
            return df

        # 计算涨跌幅
        df["change_pct"] = np.where(
            df["pre_close"] > 0,
            (df["close"] - df["pre_close"]) / df["pre_close"] * 100,
            0
        )
        # 计算换手率 / 振幅
        df["turnover"] = 0.0
        df["amplitude"] = np.where(
            df["pre_close"] > 0,
            (df["high"] - df["low"]) / df["pre_close"] * 100,
            0
        )

        # 从 DB 补 PE/PB/市值（基本面变化慢，缓存可用）
        df = self._enrich_fundamentals(df)

        logger.info(f"sina: 获取股票列表 {len(df)} 只")
        self.db.cache_put(cache_key, "stock_list", df, self._source, self._cache_ttl)
        return df

    def _enrich_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        """从本地DB补充PE/PB/市值。"""
        try:
            from database import get_market_db
            mkt = get_market_db()
            c = mkt.conn
            latest = c.execute("SELECT MAX(date) FROM daily_price WHERE code LIKE '%.SZ' OR code LIKE '%.SH'").fetchone()[0]
            # 尝试从 market_data_cache 获取最新的 daily_snapshot
            snap_key = f"daily_snapshot_{latest}"
            snap = self.db.cache_get(snap_key) if hasattr(self.db, 'cache_get') else None
            if snap is not None and len(snap) > 0:
                snap["code"] = snap["code"].astype(str).str.zfill(6)
                for col in ["pe", "pb", "market_cap"]:
                    if col in snap.columns:
                        m = snap.set_index("code")[col].to_dict()
                        df[col] = df["code"].map(m).fillna(0)
            mkt.close()
        except Exception:
            df["pe"] = 0
            df["pb"] = 0
            df["market_cap"] = 0

        if "pe" not in df.columns:
            df["pe"] = 0
        if "pb" not in df.columns:
            df["pb"] = 0
        if "market_cap" not in df.columns:
            df["market_cap"] = 0
        return df

    # ---- K线 ----

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> pd.DataFrame:
        """获取个股K线（从本地DB，新浪历史接口复杂）。"""
        try:
            from database import get_market_db
            mkt = get_market_db()
            c = mkt.conn
            sc = _sina_code(code).replace("sh", "").replace("sz", "")
            # 从 daily_price 读取最近 days 天的数据
            rows = c.execute(
                "SELECT date, open, close, high, low, volume FROM daily_price "
                "WHERE code LIKE ? ORDER BY date DESC LIMIT ?",
                (f"{sc}%", days)
            ).fetchall()
            mkt.close()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows[::-1], columns=["date", "open", "close", "high", "low", "volume"])
            return df
        except Exception:
            return pd.DataFrame()

    def get_index_kline(self, code: str, days: int = 60) -> pd.DataFrame:
        """获取指数K线。新浪指数历史走缓存。"""
        sina_code = INDEX_MAP.get(code, _sina_code(code))
        cache_key = f"sina_idx_{code}_{days}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        # 新浪日报指数接口一次只能取当天
        # 折中方案：从 DB 读取指数日线
        try:
            from database import get_market_db
            mkt = get_market_db()
            c = mkt.conn
            sc = code.zfill(6)
            rows = c.execute(
                "SELECT date, close FROM daily_price WHERE code LIKE ? ORDER BY date DESC LIMIT ?",
                (f"{sc}%", days)
            ).fetchall()
            mkt.close()
            if rows:
                df = pd.DataFrame(rows[::-1], columns=["date", "close"])
                df["close"] = df["close"].astype(float)
                self.db.cache_put(cache_key, "index_kline", df, self._source, self._cache_ttl)
                return df
        except Exception:
            pass

        # 兜底：新浪当日数据
        try:
            url = f"http://hq.sinajs.cn/list={sina_code}"
            req = urllib.request.Request(url, headers=HEADERS)
            r = urllib.request.urlopen(req, timeout=5)
            text = r.read().decode("gbk")
            fields = text.split('"')[1].split(",") if '"' in text else []
            if len(fields) > 3:
                return pd.DataFrame([{
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "close": float(fields[3]) if len(fields) > 3 else 0,
                }])
        except Exception:
            pass

        return pd.DataFrame()

    # ---- 基本面 ----

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """获取基本面数据（PE/PB/市值）。"""
        stock_list = self.get_stock_list()
        if stock_list.empty:
            return pd.DataFrame()
        return stock_list[stock_list["code"].isin(codes)]

    # ---- 板块 ----

    def get_sector_mapping(self) -> dict[str, str]:
        """代码→板块映射（从DB快照取）。"""
        cache_key = "sina_sector_map"
        cached = self.db.cache_get(cache_key) if hasattr(self.db, 'cache_get') else None
        if cached is not None:
            return dict(zip(cached["code"], cached["sector"]))

        try:
            from database import get_market_db
            mkt = get_market_db()
            c = mkt.conn
            latest = c.execute("SELECT MAX(date) FROM daily_price").fetchone()[0]
            snap_key = f"daily_snapshot_{latest}"
            snap = self.db.cache_get(snap_key) if hasattr(self.db, 'cache_get') else None
            mkt.close()
            if snap is not None and "sector" in snap.columns:
                result = dict(zip(snap["code"].astype(str).str.zfill(6), snap["sector"]))
                return result
        except Exception:
            pass
        return {}

    def get_sector_list(self) -> pd.DataFrame:
        """获取板块行情列表（DB读取）。"""
        cache_key = "sina_sector_list"
        cached = self.db.cache_get(cache_key) if hasattr(self.db, 'cache_get') else None
        if cached is not None:
            return cached
        return pd.DataFrame()


# ---- 全局单例 ----
_provider: Optional[SinaProvider] = None


def get_sina() -> SinaProvider:
    global _provider
    if _provider is None:
        _provider = SinaProvider()
    return _provider
