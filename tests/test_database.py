"""database 层回归测试: 修复 stock_picks 持久化 + runs 表迁移。

根因: 旧 stock_picks 以 run_id 为 PRIMARY KEY, 但一次运行的多只股票共享同一 run_id,
executemany 第2行起撞 PK -> IntegrityError -> 整批回滚 -> stock_picks 永为 0 行,
连带 factor_scores/holdings_snapshot(③④) 也写不进。
"""
import sqlite3

import pandas as pd
import pytest

import database as dbmod


def _categories():
    return {
        "②A 质量榜": pd.DataFrame([
            {"code": "600519", "name": "贵州茅台", "composite_score": 88.5, "sector": "白酒"},
            {"code": "000858", "name": "五粮液", "composite_score": 85.0, "sector": "白酒"},
        ]),
        "②B 短线反弹": pd.DataFrame([
            {"code": "300750", "name": "宁德时代", "composite_score": 80.0, "sector": "电池"},
        ]),
    }


def test_save_run_results_persists_multi_row(tmp_path):
    """一次运行多只股票共享 run_id, 应全部持久化(无 PK 冲突)。"""
    db = dbmod.StockDB(str(tmp_path / "t.db"))
    rid = db.save_run_results(
        {"regime": {"regime": "震荡", "position_cap": 0.5},
         "l2_filtered_count": 3, "elapsed_seconds": 9},
        _categories(), "pre_market",
    )
    con = sqlite3.connect(str(tmp_path / "t.db"))
    n_picks = con.execute("SELECT COUNT(*) FROM stock_picks").fetchone()[0]
    n_runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n_picks == 3, n_picks
    assert n_runs == 1, n_runs
    assert rid >= 1

    # factor_scores 经 FK->runs(run_id) 可写
    fs = pd.DataFrame([{
        "code": "600519", "name": "贵州茅台", "composite_score": 88.5,
        "sector": "白酒", "roe": 30.0,
    }])
    assert db.save_factor_scores(rid, fs) == 1
    assert con.execute("SELECT COUNT(*) FROM factor_scores").fetchone()[0] == 1
    con.close()


def test_save_run_results_separate_runs(tmp_path):
    """两次运行应得到不同 run_id, 各自独立。"""
    db = dbmod.StockDB(str(tmp_path / "t.db"))
    r1 = db.save_run_results(
        {"regime": {"regime": "震荡"}, "l2_filtered_count": 1, "elapsed_seconds": 1},
        _categories(), "pre_market",
    )
    r2 = db.save_run_results(
        {"regime": {"regime": "多头"}, "l2_filtered_count": 1, "elapsed_seconds": 1},
        _categories(), "post_market",
    )
    assert r1 != r2
    con = sqlite3.connect(str(tmp_path / "t.db"))
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(DISTINCT run_id) FROM stock_picks").fetchone()[0] == 2
    con.close()


def test_migration_old_schema_preserves_data(tmp_path):
    """旧 schema(run_id 为 PK) 的库打开时应自动迁移: 数据不丢 + 新 schema(id 列)。"""
    p = str(tmp_path / "old.db")
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE stock_picks (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, code TEXT, name TEXT,
        category TEXT, composite_score REAL, sector TEXT, regime TEXT,
        position_cap REAL, l2_filtered INTEGER, elapsed_sec REAL, session_type TEXT)""")
    con.execute("INSERT INTO stock_picks VALUES (1,'2026-08-20','600519','茅台','②A',88,'白酒','震荡',0.5,3,10,'pre_market')")
    con.execute("INSERT INTO stock_picks VALUES (2,'2026-08-20','000858','五粮液','②A',85,'白酒','震荡',0.5,3,10,'pre_market')")
    con.commit()
    con.close()

    db = dbmod.StockDB(p)  # 触发迁移
    con = sqlite3.connect(p)
    assert con.execute("SELECT COUNT(*) FROM stock_picks").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    cols = [r[1] for r in con.execute("PRAGMA table_info(stock_picks)").fetchall()]
    assert "id" in cols
    con.close()
    db.close()
