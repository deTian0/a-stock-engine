"""
kline_cache.py - K线数据本地缓存管理

负责K线数据的本地存储、读取和过期检查。
避免重复请求 westock-data CLI。
"""

import json
import time
import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class KlineCache:
    """K线数据缓存管理器。"""

    def __init__(self, cache_dir: str = "data_cache/kline"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, code: str, freq: str = "daily") -> Path:
        return self.cache_dir / f"{code}_{freq}.json"

    def get(self, code: str, freq: str = "daily",
            max_age_hours: int = 12) -> Optional[pd.DataFrame]:
        """
        读取缓存的K线数据。
        返回 None 表示缓存不存在或已过期。
        """
        path = self._path(code, freq)
        if not path.exists():
            return None

        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > max_age_hours * 3600:
            logger.debug(f"K线缓存过期: {code} ({freq})")
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
            return df
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"读取K线缓存失败 {code}: {e}")
            return None

    def put(self, code: str, df: pd.DataFrame, freq: str = "daily") -> None:
        """写入K线数据到缓存（原子操作，防并发竞态）。"""
        path = self._path(code, freq)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            df_to_save = df.copy()
            if "date" in df_to_save.columns and hasattr(df_to_save["date"].dtype, "tz"):
                df_to_save["date"] = df_to_save["date"].dt.tz_localize(None)
            records = df_to_save.to_dict(orient="records")
            data = json.dumps(records, ensure_ascii=False, default=str)
            # 原子写入：先写临时文件，再 rename
            tmp_path.write_text(data, encoding="utf-8")
            os.replace(tmp_path, path)  # Windows 上也是原子的
            logger.debug(f"K线缓存写入: {code} ({freq}), {len(df)} 条")
        except (OSError, TypeError) as e:
            logger.warning(f"写入K线缓存失败 {code}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def clear(self, code: Optional[str] = None) -> None:
        """清除缓存。code=None 清除全部。"""
        if code:
            for freq in ["daily", "weekly", "monthly"]:
                p = self._path(code, freq)
                if p.exists():
                    p.unlink()
        else:
            for p in self.cache_dir.glob("*.json"):
                p.unlink()
        logger.info(f"K线缓存已清除: {'全部' if code is None else code}")

    def list_cached(self) -> list[str]:
        """列出所有已缓存的股票代码。"""
        codes = set()
        for p in self.cache_dir.glob("*.json"):
            # 文件名格式: {code}_{freq}.json
            stem = p.stem
            code = stem.rsplit("_", 1)[0]
            codes.add(code)
        return sorted(codes)
