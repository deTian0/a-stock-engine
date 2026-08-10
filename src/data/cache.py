"""
cache.py - 数据持久化缓存层

所有远程数据调用自动先查本地缓存，命中直接返回，未命中才调 API。
缓存存入 SQLite 的 market_data_cache 表，支持 TTL 过期机制。

用法:
    from data.cache import cached
    @cached("stock_list", ttl_hours=1)
    def get_stock_list():
        return api_call()
"""

import logging
import json
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """持久化数据缓存管理器。"""

    def __init__(self, db_path: str = "data_cache/cache.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_cache (
                cache_key   TEXT PRIMARY KEY,
                data_type   TEXT NOT NULL,
                data_json   TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now','localtime')),
                expires_at  TEXT NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_type ON data_cache(data_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expiry ON data_cache(expires_at)")
        self.conn.commit()

    def get(self, cache_key: str) -> pd.DataFrame | None:
        """读取缓存，过期返回 None。"""
        row = self.conn.execute(
            "SELECT data_json, expires_at FROM data_cache WHERE cache_key=?",
            (cache_key,)
        ).fetchone()
        if not row:
            return None
        data_json, expires_at = row
        if expires_at < datetime.now().strftime("%Y-%m-%d %H:%M:%S"):
            self.conn.execute("DELETE FROM data_cache WHERE cache_key=?", (cache_key,))
            self.conn.commit()
            return None
        try:
            records = json.loads(data_json)
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning(f"缓存解析失败 {cache_key}: {e}")
            return None

    def put(self, cache_key: str, data_type: str, df: pd.DataFrame, ttl_hours: float):
        """写入缓存，设置过期时间。"""
        if df is None or len(df) == 0:
            return
        expires_at = (datetime.now() + timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
        records = df.head(5000).to_dict(orient="records")  # 最多缓存 5000 行
        data_json = json.dumps(records, ensure_ascii=False, default=str)
        self.conn.execute(
            "INSERT OR REPLACE INTO data_cache (cache_key, data_type, data_json, expires_at) VALUES (?,?,?,?)",
            (cache_key, data_type, data_json, expires_at)
        )
        self.conn.commit()

    def stats(self) -> dict:
        """缓存统计。"""
        total = self.conn.execute("SELECT COUNT(*) FROM data_cache").fetchone()[0]
        expired = self.conn.execute(
            "SELECT COUNT(*) FROM data_cache WHERE expires_at < datetime('now','localtime')"
        ).fetchone()[0]
        return {"total": total, "expired": expired, "active": total - expired}

    def cleanup(self):
        """清理过期缓存。"""
        n = self.conn.execute(
            "DELETE FROM data_cache WHERE expires_at < datetime('now','localtime')"
        ).rowcount
        self.conn.commit()
        if n > 0:
            logger.info(f"清理过期缓存: {n} 条")
        return n

    def close(self):
        self.conn.close()


# 全局单例
_cache: DataCache | None = None


def get_cache() -> DataCache:
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache
