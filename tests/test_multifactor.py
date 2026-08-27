"""multifactor 单元测试：categorize / assess_regime（隔离 IO）。

score_l4 / run 等重 IO 方法不在此做单元断言（已由集成测试与手动 run 覆盖）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import multifactor as mf


def _engine(monkeypatch, config_dict):
    monkeypatch.setattr(mf, "get_cli", lambda *a, **k: object())
    monkeypatch.setattr(mf, "LocalPriceLoader", lambda *a, **k: object())
    return mf.MultiFactorEngine(config_dict=config_dict)


def test_categorize_splits_lists(monkeypatch):
    cfg = {"data_source": {"cache_dir": "data_cache"},
           "output": {"quality_top_n": 2, "short_term_top_n": 1,
                      "watchlist_top_n": 5, "min_composite_score": 0}}
    eng = _engine(monkeypatch, cfg)
    l4 = pd.DataFrame({
        "code": ["A", "B", "C", "D"],
        "name": ["a", "b", "c", "d"],
        "composite_score": [90.0, 80.0, 70.0, 60.0],
        "entry_ok": [True, False, True, False],
    })
    out = eng.categorize(l4, pd.DataFrame(), {})
    assert len(out["②A_质量榜"]) == 2
    assert len(out["②B_短线榜"]) == 1
    # ②B 优先 entry_ok
    assert out["②B_短线榜"].iloc[0]["code"] == "A" or out["②B_短线榜"].iloc[0]["code"] == "C"
    assert len(out["③C_观察名单"]) == 2  # 余下 2 只进观察


def test_categorize_min_score_filter(monkeypatch):
    cfg = {"data_source": {"cache_dir": "data_cache"},
           "output": {"quality_top_n": 10, "short_term_top_n": 5,
                      "watchlist_top_n": 5, "min_composite_score": 75}}
    eng = _engine(monkeypatch, cfg)
    l4 = pd.DataFrame({
        "code": ["A", "B", "C"],
        "name": ["a", "b", "c"],
        "composite_score": [90.0, 70.0, 60.0],  # B/C 低于 75 被过滤
        "entry_ok": [False, False, False],
    })
    out = eng.categorize(l4, pd.DataFrame(), {})
    # 仅 A 通过阈值
    assert set(out["②A_质量榜"]["code"]) == {"A"}


def test_assess_regime_bull(monkeypatch):
    cfg = {
        "data_source": {"cache_dir": "data_cache"},
        "env_regime": {
            "indices": [{"code": "000001", "name": "上证"}],
            "ma_short": 20, "ma_long": 60,
            "regime_thresholds": {"bull_above_ma_ratio": 0.5, "bear_below_ma_ratio": 0.5},
            "position_caps": {"多头": 0.8, "空头": 0.2, "震荡": 0.5},
        },
        "output": {},
    }
    eng = _engine(monkeypatch, cfg)
    dates = pd.date_range("2026-01-01", periods=65)
    df = pd.DataFrame({
        "date": dates,
        "close": list(np.linspace(3000, 3500, 65)),  # 收盘高于 MA60
    })

    class FakeCli:
        def get_index_kline(self, code, days=70):
            return df

    eng.cli = FakeCli()
    res = eng.assess_regime()
    assert res["regime"] == "多头"
    assert res["position_cap"] == 0.8
    assert res["indices"]["000001"]["above_ma"] is True


def test_assess_regime_bear(monkeypatch):
    cfg = {
        "data_source": {"cache_dir": "data_cache"},
        "env_regime": {
            "indices": [{"code": "000001", "name": "上证"}],
            "ma_short": 20, "ma_long": 60,
            "regime_thresholds": {"bull_above_ma_ratio": 0.5, "bear_below_ma_ratio": 0.5},
            "position_caps": {"多头": 0.8, "空头": 0.2, "震荡": 0.5},
        },
        "output": {},
    }
    eng = _engine(monkeypatch, cfg)
    dates = pd.date_range("2026-01-01", periods=65)
    df = pd.DataFrame({
        "date": dates,
        "close": list(np.linspace(3500, 3000, 65)),  # 收盘低于 MA60
    })

    class FakeCli:
        def get_index_kline(self, code, days=70):
            return df

    eng.cli = FakeCli()
    res = eng.assess_regime()
    assert res["regime"] == "空头"
    assert res["position_cap"] == 0.2


def test_select_etfs_avoids_stale_db_and_uses_live(monkeypatch, caplog):
    """回归测试：DB 动量快照过期时，必须回退实时行情，绝不输出冻结的虚假动量。

    复现 2026-08 线上 bug：market.db 的 ETF daily_price 停留在 2026-03-13，
    旧逻辑直接采用冻结动量 => 每日早盘 ETF 推荐完全一样。
    """
    import logging
    cfg = {"data_source": {"cache_dir": "data_cache"}, "output": {}}
    eng = _engine(monkeypatch, cfg)

    # 模拟 DB 返回冻结快照（as_of 远超 10 天）
    stale = {
        "515790": {"momentum_20d": 6.4, "momentum_60d": -1.0, "as_of": "2026-03-13"},
        "510500": {"momentum_20d": 1.1, "momentum_60d": 2.0, "as_of": "2026-03-13"},
    }
    monkeypatch.setattr(eng, "_batch_calc_momentum", lambda codes: stale)

    # 模拟实时行情：返回近期序列，动量随 code 不同且非冻结值
    class FakeLoader:
        def get_price(self, code, days=60, adjust="qfq"):
            n = 62
            base = {"515790": 1.0, "510500": 2.0}.get(code, 1.5)
            closes = [base * (1 + 0.001 * i) for i in range(n)]  # 20日动量 ≈ +2.0%
            return pd.DataFrame({
                "date": pd.date_range("2026-06-01", periods=n),
                "close": closes,
                "amount": [1e8] * n,
            })

    eng.price_loader = FakeLoader()

    with caplog.at_level(logging.WARNING):
        res = eng.select_etfs(top_n=8)

    assert not res.empty, "实时回退应至少选出部分 ETF"
    # 关键断言：不得出现冻结的虚假动量值
    assert 6.4 not in res["momentum_20d"].values
    assert 1.1 not in res["momentum_20d"].values
    # 实时动量应为正（≈2%）而非冻结值
    assert (res["momentum_20d"] > 0).any()
    assert any("过期" in r.message for r in caplog.records), "应记录 DB 数据过期告警"


def test_select_etfs_skips_when_live_also_fails(monkeypatch, caplog):
    """当 DB 冻结且实时获取也失败时，应跳过而非输出冻结值。"""
    import logging
    cfg = {"data_source": {"cache_dir": "data_cache"}, "output": {}}
    eng = _engine(monkeypatch, cfg)

    stale = {"515790": {"momentum_20d": 6.4, "momentum_60d": -1.0, "as_of": "2026-03-13"}}
    monkeypatch.setattr(eng, "_batch_calc_momentum", lambda codes: stale)

    class EmptyLoader:
        def get_price(self, code, days=60, adjust="qfq"):
            return pd.DataFrame()  # 实时失败

    eng.price_loader = EmptyLoader()

    with caplog.at_level(logging.WARNING):
        res = eng.select_etfs(top_n=8)

    assert res.empty, "实时失败时应不输出任何冻结 ETF"
    assert any("不可用" in r.message for r in caplog.records)



import sqlite3
import database as dbmod
import logging


class _MemStockDB(dbmod.StockDB):
    """用内存连接覆盖只读 conn property，测试真实 delete SQL。"""
    def __init__(self, con):
        self._con = con

    @property
    def conn(self):
        return self._con


def _memory_market_db():
    """构造一个仅含 daily_price 的内存 StockDB，用于测试真实 SQL。"""
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE daily_price ("
        "code TEXT NOT NULL, date TEXT NOT NULL, close REAL, pct_chg REAL, "
        "vol REAL, amount REAL, PRIMARY KEY(code,date))"
    )
    return _MemStockDB(con)


def test_delete_prices_for_codes_removes_all_forms():
    mdb = _memory_market_db()
    con = mdb.conn
    con.executemany(
        "INSERT INTO daily_price VALUES (?,?,?,?,?,?)",
        [
            ("510500", "2026-03-13", 1.0, 0, 0, 0),
            ("510500.SH", "2026-03-13", 1.0, 0, 0, 0),
            ("510500.SZ", "2026-03-13", 1.0, 0, 0, 0),
            ("159915", "2026-03-13", 1.0, 0, 0, 0),
            ("000001.SZ", "2026-08-20", 2.0, 0, 0, 0),  # 无关股票，不应被删
        ],
    )
    n = mdb.delete_prices_for_codes(["510500", "159915"])
    remaining = {r[0] for r in con.execute("SELECT code FROM daily_price")}
    assert remaining == {"000001.SZ"}, remaining
    assert n == 4


def test_refresh_etf_daily_prices_deletes_old_and_upserts_with_suffix(monkeypatch):
    deleted = []
    inserted = []

    class FakeMdb:
        def delete_prices_for_codes(self, codes):
            deleted.extend(codes)
            return len(codes) * 3

        def bulk_insert_prices(self, rows):
            inserted.extend(rows)
            return len(rows)

    monkeypatch.setattr(dbmod, "get_market_db", lambda *a, **k: FakeMdb())

    dates = pd.date_range("2026-08-01", periods=65).strftime("%Y-%m-%d").tolist()

    class FakeLoader:
        def get_price(self, code, days=65, adjust="qfq"):
            return pd.DataFrame(
                {
                    "date": dates,
                    "close": [1.0 + i * 0.01 for i in range(65)],
                    "change_pct": [0.1] * 65,
                    "volume": [1000.0] * 65,
                    "amount": [1e6] * 65,
                }
            )

    res = mf.refresh_etf_daily_prices(FakeLoader(), days=65)
    codes = [c for c, _ in mf.WELL_KNOWN_ETFS]
    assert res["refreshed"] == len(codes)
    assert res["failed"] == []
    # 刷新以纯码列表传入删除（.SZ/.SH 三态展开在 delete_prices_for_codes 内部，由 test 1 覆盖）
    assert set(deleted) == set(codes)
    # 写入使用正确交易所后缀，且均为带后缀形态（杜绝新旧行合并）
    ins_codes = {r[0] for r in inserted}
    for c in codes:
        sfx = "SH" if c.startswith(("5", "6")) else "SZ"
        assert f"{c}.{sfx}" in ins_codes, (c, sfx, ins_codes)
    assert all("." in r[0] for r in inserted), "不应写入纯码形态"


def test_refresh_etf_daily_prices_records_failed(monkeypatch):
    deleted = []
    inserted = []

    class FakeMdb:
        def delete_prices_for_codes(self, codes):
            deleted.extend(codes)
            return 0

        def bulk_insert_prices(self, rows):
            inserted.extend(rows)
            return len(rows)

    monkeypatch.setattr(dbmod, "get_market_db", lambda *a, **k: FakeMdb())

    class FlakyLoader:
        def get_price(self, code, days=65, adjust="qfq"):
            if code == "515790":
                return pd.DataFrame()  # 模拟该 ETF 拉取失败
            dates = pd.date_range("2026-08-01", periods=65).strftime("%Y-%m-%d").tolist()
            return pd.DataFrame(
                {
                    "date": dates,
                    "close": [1.0] * 65,
                    "change_pct": [0.0] * 65,
                    "volume": [1.0] * 65,
                    "amount": [1.0] * 65,
                }
            )

    res = mf.refresh_etf_daily_prices(FlakyLoader(), days=65)
    assert res["failed"] == ["515790"], res
    assert res["refreshed"] == len(mf.WELL_KNOWN_ETFS) - 1
