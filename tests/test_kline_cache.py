"""kline_cache 单元测试：缓存生命周期（正常/边界/异常）。

使用 pytest tmp_path 隔离，无真实网络/磁盘污染。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kline_cache import KlineCache


def _sample_df(n=10):
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "close": list(range(n)),
        "volume": [100] * n,
    })


def test_put_then_get_roundtrip(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    df = _sample_df()
    c.put("600519", df)
    got = c.get("600519")
    assert got is not None
    assert len(got) == 10
    assert list(got["close"]) == list(range(10))


def test_get_missing_returns_none(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    assert c.get("000001") is None


def test_get_expired_returns_none(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    c.put("600519", _sample_df())
    # 把文件 mtime 改到 100 小时前 -> 超过默认 12h
    p = c._path("600519")
    old = p.stat().st_mtime - 100 * 3600
    import os
    os.utime(p, (old, old))
    assert c.get("600519", max_age_hours=12) is None


def test_clear_single_code(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    c.put("600519", _sample_df())
    c.put("000001", _sample_df())
    c.clear("600519")
    assert c.get("600519") is None
    assert c.get("000001") is not None


def test_clear_all(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    c.put("600519", _sample_df())
    c.put("000001", _sample_df())
    c.clear()
    assert c.list_cached() == []


def test_list_cached(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    c.put("600519", _sample_df())
    c.put("000001", _sample_df())
    assert c.list_cached() == ["000001", "600519"]


def test_put_corrupt_json_does_not_crash(tmp_path):
    c = KlineCache(cache_dir=str(tmp_path / "k"))
    # 直接塞一个非法 json 文件
    (tmp_path / "k" / "bad_daily.json").write_text("{not valid", encoding="utf-8")
    assert c.get("bad") is None  # 解析失败返回 None 而非抛异常
