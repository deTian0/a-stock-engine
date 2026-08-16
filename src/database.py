"""
database.py - SQLite 数据库层

提供选股系统的数据持久化能力。
4张核心表：
  stock_picks       — 每次运行的选股结果
  t2_verifications  — T+2 验证逐笔记录
  holdings_snapshot — 每日持仓快照
  factor_scores     — 每只股票每次运行的因子评分明细

用法:
    from database import StockDB
    db = StockDB("data_cache/a-stock-engine.db")

    run_id = db.save_run_results(results, categories)
    db.save_factor_scores(run_id, enriched_df, l4_df)
    db.save_holdings_snapshot(holdings_data)
    db.save_t2_verification(pick_date, verifications_list)

    history = db.get_run_history(days=30)
    stats = db.get_t2_stats(days=30)
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from io import StringIO

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4


class StockDB:
    """选股系统 SQLite 数据库。"""

    def __init__(self, db_path: str = "data_cache/a-stock-engine.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def commit(self):
        """显式提交事务。"""
        if self._conn is not None:
            self._conn.commit()

    def _init_schema(self):
        """建表（如果不存在）。"""
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")

        row = c.execute("SELECT version FROM schema_version").fetchone()
        current = row["version"] if row else 0

        if current >= SCHEMA_VERSION:
            return

        logger.info("初始化数据库表结构...")

        c.executescript("""
        CREATE TABLE IF NOT EXISTS stock_picks (
            run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT    NOT NULL,
            code         TEXT    NOT NULL,
            name         TEXT,
            category     TEXT    NOT NULL,
            composite_score REAL,
            sector       TEXT,
            regime       TEXT,
            position_cap REAL,
            l2_filtered  INTEGER,
            elapsed_sec  REAL,
            session_type TEXT    DEFAULT 'pre_market',
            created_at   TEXT    DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS t2_verifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            code         TEXT    NOT NULL,
            name         TEXT,
            pick_date    TEXT    NOT NULL,
            t0_close     REAL,
            t2_close     REAL,
            return_pct   REAL,
            status       TEXT,
            verified_at  TEXT    DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS holdings_snapshot (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    NOT NULL,
            code          TEXT    NOT NULL,
            name          TEXT,
            shares        INTEGER,
            cost_price    REAL,
            current_price REAL,
            cost_value    REAL,
            market_value  REAL,
            hold_return_pct REAL,
            composite_score REAL,
            sector        TEXT,
            created_at    TEXT    DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS factor_scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL,
            code            TEXT    NOT NULL,
            name            TEXT,
            composite_score REAL,
            pe              REAL,
            pb              REAL,
            roe             REAL,
            gross_margin    REAL,
            debt_ratio      REAL,
            revenue_growth  REAL,
            profit_growth   REAL,
            momentum_20d    REAL,
            momentum_60d    REAL,
            market_cap      REAL,
            sector          TEXT,
            rsi             REAL,
            kdj_k           REAL,
            kdj_d           REAL,
            kdj_j           REAL,
            ma5_slope       REAL,
            ma10_slope      REAL,
            volume_ratio    REAL,
            short_signal    TEXT,
            created_at      TEXT    DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (run_id) REFERENCES stock_picks(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_stock_picks_date   ON stock_picks(date);
        CREATE INDEX IF NOT EXISTS idx_stock_picks_code   ON stock_picks(code);
        CREATE INDEX IF NOT EXISTS idx_t2_pick_date       ON t2_verifications(pick_date);
        CREATE INDEX IF NOT EXISTS idx_t2_code            ON t2_verifications(code);
        CREATE INDEX IF NOT EXISTS idx_holdings_date      ON holdings_snapshot(date);
        CREATE INDEX IF NOT EXISTS idx_factor_run         ON factor_scores(run_id);
        CREATE INDEX IF NOT EXISTS idx_factor_code        ON factor_scores(code);

        CREATE TABLE IF NOT EXISTS market_data_cache (
            cache_key     TEXT PRIMARY KEY,
            data_type     TEXT    NOT NULL,
            data_json     TEXT    NOT NULL,
            source        TEXT,
            rows_count    INTEGER,
            created_at    TEXT    DEFAULT (datetime('now','localtime')),
            expires_at    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_market_type ON market_data_cache(data_type);

        -- 板块轮动追踪（v3 新增）：记录每轮选股窗口
        CREATE TABLE IF NOT EXISTS sector_rotation_tracking (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            window_start  TEXT    NOT NULL,
            window_end    TEXT    NOT NULL,
            session_type  TEXT    NOT NULL,
            sector_name   TEXT    NOT NULL,
            code          TEXT    NOT NULL,
            name          TEXT,
            asset_type    TEXT    DEFAULT 'stock',
            score         REAL,
            close_price   REAL,
            pick_rank     INTEGER,
            created_at    TEXT    DEFAULT (datetime('now','localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_rot_window   ON sector_rotation_tracking(window_start, window_end);
        CREATE INDEX IF NOT EXISTS idx_rot_sector   ON sector_rotation_tracking(sector_name, session_type);
        CREATE INDEX IF NOT EXISTS idx_rot_code     ON sector_rotation_tracking(code);

        -- 归一化价格表（v3 新增）：code+date 复合主键，只存回测必需列
        CREATE TABLE IF NOT EXISTS daily_price (
            code          TEXT    NOT NULL,
            date          TEXT    NOT NULL,
            close         REAL,
            pct_chg       REAL,
            vol           REAL,
            amount        REAL,
            PRIMARY KEY (code, date)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_dp_date ON daily_price(date);

        -- 累计命中次数（v3 新增）：追踪每只股票被选中的历史
        CREATE TABLE IF NOT EXISTS pick_frequency (
            code          TEXT    NOT NULL,
            session_type  TEXT    NOT NULL,
            total_hits    INTEGER DEFAULT 0,
            last_hit_date TEXT,
            first_hit_date TEXT,
            PRIMARY KEY (code, session_type)
        );
        CREATE INDEX IF NOT EXISTS idx_pfreq_hits ON pick_frequency(total_hits DESC);

        -- 持久化基本面表（v4 新增）：全市场截面基本面快照，code 主键
        -- 估值来自 daily_basic（total_mv/circ_mv 已转元）；财务来自 fina_indicator（period 过滤）
        CREATE TABLE IF NOT EXISTS fundamentals (
            code            TEXT PRIMARY KEY,
            name            TEXT,
            industry        TEXT,
            area            TEXT,
            list_date       TEXT,
            market          TEXT,
            pe              REAL,
            pe_ttm          REAL,
            pb              REAL,
            ps              REAL,
            ps_ttm          REAL,
            dv_ratio        REAL,
            total_mv        REAL,
            circ_mv         REAL,
            roe             REAL,
            roa             REAL,
            gross_margin    REAL,
            debt_ratio      REAL,
            revenue_growth  REAL,
            profit_growth   REAL,
            dedt_net_profit REAL,
            eps             REAL,
            bps             REAL,
            report_period   TEXT,
            valuation_date  TEXT,
            updated_at      TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_fund_code ON fundamentals(code);

        -- PIT 财务报表(时点正确): 每只股票每个报告期一行, 含公告日 ann_date
        -- 回测时只取 ann_date <= 回测日 的最近报告期, 从根上杜绝「用未来财报」的前视
        CREATE TABLE IF NOT EXISTS fundamentals_pit (
            code           TEXT    NOT NULL,
            end_date       TEXT    NOT NULL,   -- 报告期 (YYYYMMDD)
            ann_date       TEXT,               -- 公告日 (YYYYMMDD), PIT 闸口
            roe            REAL,
            roa            REAL,
            gross_margin   REAL,
            debt_ratio     REAL,
            revenue_growth REAL,
            profit_growth  REAL,
            eps            REAL,
            eps_ttm        REAL,
            bps            REAL,
            report_type    TEXT,
            PRIMARY KEY (code, end_date)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_pit_ann ON fundamentals_pit(code, ann_date);

        -- PIT 估值(daily_basic, 时点正确): 每只股票每个交易日一行
        -- 回测时取 trade_date = 回测日 的 pe/pb/total_mv, 杜绝「用最新估值」的前视
        CREATE TABLE IF NOT EXISTS daily_basic_pit (
            code          TEXT    NOT NULL,
            trade_date    TEXT    NOT NULL,
            pe            REAL,
            pe_ttm        REAL,
            pb            REAL,
            ps            REAL,
            ps_ttm        REAL,
            dv_ratio      REAL,
            total_mv      REAL,
            circ_mv       REAL,
            PRIMARY KEY (code, trade_date)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_dbp_date ON daily_basic_pit(trade_date);
        """)

        c.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        c.commit()

        # v3 升级：为已有 stock_picks 表添加 session_type 列（忽略重复添加错误）
        try:
            c.execute("ALTER TABLE stock_picks ADD COLUMN session_type TEXT DEFAULT 'pre_market'")
            c.commit()
        except sqlite3.OperationalError:
            pass  # 列已存在

        # v4.14: t2_verifications 增加 category 列（区分买入类目, 支撑纯"推荐买入"胜率分桶）
        try:
            c.execute("ALTER TABLE t2_verifications ADD COLUMN category TEXT")
            c.commit()
        except sqlite3.OperationalError:
            pass  # 列已存在

        logger.info("数据库初始化完成")

    # ============================================
    #  写入方法
    # ============================================

    def save_run_results(self, results: dict, categories: dict,
                         session_type: str = "pre_market") -> int:
        """
        保存一次选股运行的完整结果。
        session_type: 'pre_market' 或 'post_market'
        返回本次运行的 run_id。
        """
        regime = results.get("regime", {})
        regime_name = regime.get("regime", "未知") if isinstance(regime, dict) else "未知"
        cap = regime.get("position_cap", 0) if isinstance(regime, dict) else 0

        today = datetime.now().strftime("%Y-%m-%d")
        c = self.conn

        run_id = c.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM stock_picks").fetchone()[0]

        rows = []
        for cat_name, cat_df in categories.items():
            if cat_df is None or len(cat_df) == 0:
                continue
            for _, row in cat_df.iterrows():
                # 标准化 code 为 6 位字符串，杜绝 '6' 与 '000006' 并存的历史 bug
                code_raw = row.get("code", "")
                code = str(code_raw).strip()
                code = code.split(".")[0] if "." in code else code
                if code.isdigit():
                    code = code.zfill(6)
                rows.append((
                    run_id, today, code,
                    str(row.get("name", code)),
                    cat_name,
                    float(row.get("composite_score", 0)) if row.get("composite_score") is not None else 0,
                    str(row.get("sector", "")),
                    regime_name, cap,
                    results.get("l2_filtered_count", 0),
                    results.get("elapsed_seconds", 0),
                    session_type,
                ))

        if rows:
            c.executemany("""
                INSERT INTO stock_picks
                (run_id, date, code, name, category, composite_score, sector,
                 regime, position_cap, l2_filtered, elapsed_sec, session_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            c.commit()

        logger.info(f"选股结果已入库: run_id={run_id}, {len(rows)} 条, session={session_type}")
        return run_id

    def save_factor_scores(self, run_id: int, enriched_df: pd.DataFrame,
                           l4_df: pd.DataFrame = None) -> int:
        """
        保存因子评分明细。
        """
        c = self.conn

        df = enriched_df.copy()
        if "code" not in df.columns:
            logger.warning("factor_scores: 数据缺少 code 列")
            return 0

        def _f(key, default=0.0):
            v = row.get(key)
            return float(v) if v is not None and pd.notna(v) else default

        def _s(key, default=""):
            v = row.get(key)
            return str(v) if v is not None and pd.notna(v) else default

        rows = []
        for _, row in df.iterrows():
            code = row.get("code", "")

            rows.append((
                run_id, code, _s("name", code),
                _f("composite_score"), _f("pe"), _f("pb"), _f("roe"),
                _f("gross_margin"), _f("debt_ratio"),
                _f("revenue_growth"), _f("profit_growth"),
                _f("momentum_20d"), _f("momentum_60d"),
                _f("market_cap"), _s("sector"),
                _f("rsi"), _f("kdj_k"), _f("kdj_d"), _f("kdj_j"),
                _f("ma5_slope"), _f("ma10_slope"), _f("volume_ratio"),
                _s("short_signal"),
            ))

        if rows:
            c.executemany("""
                INSERT INTO factor_scores
                (run_id, code, name, composite_score, pe, pb, roe,
                 gross_margin, debt_ratio, revenue_growth, profit_growth,
                 momentum_20d, momentum_60d, market_cap, sector,
                 rsi, kdj_k, kdj_d, kdj_j, ma5_slope, ma10_slope,
                 volume_ratio, short_signal)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            c.commit()

        logger.info(f"因子评分已入库: run_id={run_id}, {len(rows)} 条")
        return len(rows)

    def save_holdings_snapshot(self, holdings: dict, l4_results: pd.DataFrame = None) -> int:
        """
        保存每日持仓快照。
        holdings: config 中的 {code: {name, shares, cost_price}}
        l4_results: L4 评分结果，用于补充当前评分
        """
        if not holdings:
            return 0

        today = datetime.now().strftime("%Y-%m-%d")
        c = self.conn
        rows = []

        for code, info in holdings.items():
            if not isinstance(info, dict):
                continue
            name = info.get("name", code)
            shares = info.get("shares", 0)
            cost_price = info.get("cost_price", 0)

            cost_value = shares * cost_price
            current_price = cost_price  # 默认用成本价

            # 从 L4 结果中查找当前评分
            score = None
            sector = ""
            if l4_results is not None and len(l4_results) > 0 and "code" in l4_results.columns:
                match = l4_results[l4_results["code"] == code]
                if len(match) > 0:
                    score = match.iloc[0].get("composite_score")
                    sector = str(match.iloc[0].get("sector", ""))

            hold_return = 0.0
            market_value = cost_value
            if current_price and cost_price and cost_price > 0:
                hold_return = (current_price / cost_price - 1) * 100
                market_value = shares * current_price

            rows.append((
                today, code, str(name), shares, cost_price,
                current_price, cost_value, market_value,
                round(hold_return, 2),
                round(float(score), 2) if score is not None else None,
                sector,
            ))

        if rows:
            c.executemany("""
                INSERT INTO holdings_snapshot
                (date, code, name, shares, cost_price, current_price,
                 cost_value, market_value, hold_return_pct, composite_score, sector)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            c.commit()

        logger.info(f"持仓快照已入库: {today}, {len(rows)} 只")
        return len(rows)

    def save_t2_verification(self, pick_date: str, verifications: list[dict]) -> int:
        """
        保存 T+2 验证结果（**幂等**：按 pick_date+code+category 去重，重跑不累加）。
        verifications: list of {code, name, t0_close, t2_close, return_pct, status, category}
        category 为空时（如 mini_backtest 旧调用）默认 ""，向后兼容。
        code 统一规范为 6 位，杜绝 '6' 与 '000006' 并存。
        """
        if not verifications:
            return 0

        c = self.conn
        rows = []
        dedup_keys = set()
        for v in verifications:
            # 标准化 code -> 6 位
            code_raw = v.get("code", "")
            code = str(code_raw).strip()
            code = code.split(".")[0] if "." in code else code
            if code.isdigit():
                code = code.zfill(6)
            category = v.get("category", "")
            dedup_keys.add((pick_date, code, category))
            name = code if str(v.get("name", "")).isdigit() else v.get("name", "")
            rows.append((
                code,
                name,
                pick_date,
                v.get("t0_close"),
                v.get("t2_close"),
                v.get("return_pct"),
                v.get("status", ""),
                category,
            ))

        # 幂等：先删同键旧记录，避免重跑无限累加 / 残留去零副本
        for pk, cd, ct in dedup_keys:
            c.execute(
                "DELETE FROM t2_verifications WHERE pick_date=? AND code=? AND category=?",
                (pk, cd, ct),
            )

        c.executemany("""
            INSERT INTO t2_verifications
            (code, name, pick_date, t0_close, t2_close, return_pct, status, category)
            VALUES (?,?,?,?,?,?,?,?)
        """, rows)
        c.commit()

        logger.info(f"T+2验证已入库: {pick_date}, {len(rows)} 条（幂等去重）")
        return len(rows)

    # ============================================
    #  查询方法
    # ============================================

    def get_latest_run(self) -> Optional[dict]:
        """获取最近一次选股运行的汇总信息。"""
        c = self.conn
        row = c.execute("""
            SELECT run_id, date, regime, position_cap, l2_filtered, elapsed_sec,
                   COUNT(*) AS pick_count,
                   SUM(CASE WHEN category='②A_质量榜' THEN 1 ELSE 0 END) AS quality_count,
                   SUM(CASE WHEN category='②B_短线榜' THEN 1 ELSE 0 END) AS short_count,
                   SUM(CASE WHEN category='③C_观察名单' THEN 1 ELSE 0 END) AS watch_count
            FROM stock_picks
            GROUP BY run_id
            ORDER BY run_id DESC
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None

    def get_run_history(self, days: int = 30) -> list[dict]:
        """获取近N天的选股运行历史。"""
        c = self.conn
        rows = c.execute("""
            SELECT run_id, date, regime, position_cap, l2_filtered, elapsed_sec,
                   COUNT(*) AS pick_count,
                   AVG(composite_score) AS avg_score,
                   MAX(composite_score) AS max_score
            FROM stock_picks
            WHERE date >= date('now', ?)
            GROUP BY run_id
            ORDER BY run_id DESC
        """, (f"-{days} days",)).fetchall()
        return [dict(r) for r in rows]

    def get_run_detail(self, run_id: int) -> dict:
        """获取某次运行的完整明细。"""
        c = self.conn
        picks = c.execute(
            "SELECT * FROM stock_picks WHERE run_id=? ORDER BY category, composite_score DESC",
            (run_id,)
        ).fetchall()

        factors = c.execute(
            "SELECT * FROM factor_scores WHERE run_id=? ORDER BY composite_score DESC",
            (run_id,)
        ).fetchall()

        return {
            "picks": [dict(p) for p in picks],
            "factors": [dict(f) for f in factors],
        }

    def get_t2_stats(self, days: int = 30) -> dict:
        """获取近N天的 T+2 验证统计数据。"""
        c = self.conn
        rows = c.execute("""
            SELECT return_pct
            FROM t2_verifications
            WHERE pick_date >= date('now', ?)
              AND status = 'success'
              AND return_pct IS NOT NULL
        """, (f"-{days} days",)).fetchall()

        returns = [r["return_pct"] for r in rows]
        if not returns:
            return {"count": 0, "positive": 0, "win_rate": 0,
                    "avg_return": 0, "median_return": 0,
                    "max_return": 0, "min_return": 0, "std": 0}

        arr = np.array(returns)
        positive = int(np.sum(arr > 0))
        return {
            "count": len(arr),
            "positive": positive,
            "win_rate": round(positive / len(arr) * 100, 1),
            "avg_return": round(float(np.mean(arr)), 2),
            "median_return": round(float(np.median(arr)), 2),
            "max_return": round(float(np.max(arr)), 2),
            "min_return": round(float(np.min(arr)), 2),
            "std": round(float(np.std(arr)), 2),
        }

    def get_factor_effectiveness(self, days: int = 60) -> dict:
        """
        分析因子有效性：计算各因子与 T+2 收益率的相关性。
        返回 {factor_name: corr_coefficient}
        """
        c = self.conn
        rows = c.execute("""
            SELECT f.composite_score, f.pe, f.pb, f.roe, f.gross_margin,
                   f.debt_ratio, f.revenue_growth, f.profit_growth,
                   f.momentum_20d, f.momentum_60d, f.rsi,
                   f.volume_ratio, f.ma5_slope,
                   v.return_pct
            FROM factor_scores f
            JOIN t2_verifications v ON f.code = v.code
            WHERE f.created_at >= date('now', ?)
              AND v.status = 'success'
              AND v.return_pct IS NOT NULL
        """, (f"-{days} days",)).fetchall()

        if not rows:
            return {}

        factors = [
            "composite_score", "pe", "pb", "roe", "gross_margin",
            "debt_ratio", "revenue_growth", "profit_growth",
            "momentum_20d", "momentum_60d", "rsi",
            "volume_ratio", "ma5_slope",
        ]

        data = {}
        for factor in factors:
            data[factor] = [r[factor] for r in rows if r[factor] is not None]

        returns = [r["return_pct"] for r in rows]

        correlations = {}
        for factor, values in data.items():
            if len(values) < 10:
                correlations[factor] = None
                continue
            try:
                corr = np.corrcoef(values, returns)[0, 1]
                correlations[factor] = round(float(corr) if not np.isnan(corr) else 0, 4)
            except Exception:
                correlations[factor] = None

        return correlations

    def get_pick_performance(self, days: int = 30) -> pd.DataFrame:
        """
        获取近N天按 category 分组的选股表现汇总。
        返回 DataFrame，包含 category / count / avg_score / t2_avg_return
        """
        c = self.conn
        rows = c.execute("""
            SELECT s.category,
                   COUNT(DISTINCT s.code) AS stock_count,
                   AVG(s.composite_score) AS avg_score,
                   AVG(v.return_pct) AS t2_avg_return,
                   SUM(CASE WHEN v.return_pct > 0 THEN 1 ELSE 0 END) AS t2_positive,
                   COUNT(v.id) AS t2_count
            FROM stock_picks s
            LEFT JOIN t2_verifications v ON s.code = v.code
                AND s.date = v.pick_date
                AND v.status = 'success'
            WHERE s.date >= date('now', ?)
            GROUP BY s.category
            ORDER BY s.category
        """, (f"-{days} days",)).fetchall()

        return pd.DataFrame([dict(r) for r in rows])

    def search_picks(self, code: str = None, keyword: str = None,
                     category: str = None, limit: int = 50) -> pd.DataFrame:
        """
        搜索历史选股记录。
        code: 股票代码
        keyword: 名称关键词
        category: 分类（②A_质量榜/②B_短线榜等）
        """
        conditions = ["1=1"]
        params = []

        if code:
            conditions.append("code = ?")
            params.append(code)
        if keyword:
            conditions.append("name LIKE ?")
            params.append(f"%{keyword}%")
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = " AND ".join(conditions)
        c = self.conn
        rows = c.execute(
            f"SELECT * FROM stock_picks WHERE {where} ORDER BY date DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ============================================
    #  统计方法
    # ============================================

    def get_latest_holdings(self) -> pd.DataFrame:
        """获取最新的持仓快照。"""
        c = self.conn
        latest_date = c.execute(
            "SELECT MAX(date) FROM holdings_snapshot"
        ).fetchone()[0]

        if not latest_date:
            return pd.DataFrame()

        rows = c.execute(
            "SELECT * FROM holdings_snapshot WHERE date=? ORDER BY market_value DESC",
            (latest_date,)
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_daily_summary(self, date_str: str = None) -> dict:
        """获取某日（默认今天）的选股+持仓汇总。"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        c = self.conn

        latest_run = c.execute(
            "SELECT run_id FROM stock_picks WHERE date=? LIMIT 1",
            (date_str,)
        ).fetchone()

        holding = c.execute(
            "SELECT * FROM holdings_snapshot WHERE date=?",
            (date_str,)
        ).fetchall()

        t2 = c.execute(
            "SELECT * FROM t2_verifications WHERE pick_date=?",
            (date_str,)
        ).fetchall()

        return {
            "date": date_str,
            "run_id": latest_run["run_id"] if latest_run else None,
            "holdings_count": len(holding),
            "holdings_total_value": sum(h["market_value"] for h in holding if h["market_value"]),
            "t2_verifications": len(t2),
        }

    # ============================================
    #  行情数据缓存（供 akshare / westock 双源使用）
    # ============================================

    def cache_put(self, cache_key: str, data_type: str, data: pd.DataFrame,
                  source: str = "", ttl_hours: int = 12) -> None:
        """将行情数据缓存到 SQLite。"""
        c = self.conn
        expires = (datetime.now() + pd.Timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
        data_json = data.to_json(orient="records", force_ascii=False, date_format="iso")
        c.execute("""
            INSERT OR REPLACE INTO market_data_cache
            (cache_key, data_type, data_json, source, rows_count, expires_at)
            VALUES (?,?,?,?,?,?)
        """, (cache_key, data_type, data_json, source, len(data), expires))
        c.commit()
        logger.debug(f"行情缓存写入: {cache_key} ({data_type}), {len(data)} 行")

    def cache_get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从 SQLite 读取行情数据缓存。"""
        c = self.conn
        row = c.execute("""
            SELECT data_json, expires_at FROM market_data_cache
            WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > datetime('now','localtime'))
        """, (cache_key,)).fetchone()
        if row is None:
            return None
        try:
            df = pd.read_json(StringIO(row["data_json"]), orient="records")
            logger.debug(f"行情缓存命中: {cache_key}, {len(df)} 行")
            return df
        except (ValueError, KeyError) as e:
            logger.warning(f"行情缓存解析失败 {cache_key}: {e}")
            return None

    def cache_get_batch(self, data_type: str, keys: list[str]) -> dict[str, pd.DataFrame]:
        """批量读取缓存数据。返回 {key: DataFrame}"""
        if not keys:
            return {}
        c = self.conn
        placeholders = ",".join("?" for _ in keys)
        rows = c.execute(f"""
            SELECT cache_key, data_json FROM market_data_cache
            WHERE cache_key IN ({placeholders})
              AND (expires_at IS NULL OR expires_at > datetime('now','localtime'))
        """, keys).fetchall()
        results = {}
        for row in rows:
            try:
                results[row["cache_key"]] = pd.read_json(StringIO(row["data_json"]), orient="records")
            except (ValueError, KeyError):
                pass
        return results

    def cache_clear_expired(self) -> int:
        """清理过期缓存。返回清理条数。"""
        c = self.conn
        c.execute("DELETE FROM market_data_cache WHERE expires_at <= datetime('now','localtime')")
        deleted = c.rowcount
        c.commit()
        if deleted > 0:
            logger.info(f"清理过期行情缓存: {deleted} 条")
        return deleted

    # ============================================
    #  板块轮动追踪（v3 新增）
    # ============================================

    def save_rotation_picks(self, picks: list[dict], window_start: str,
                            window_end: str, session_type: str) -> int:
        """保存一轮板块选股结果。"""
        c = self.conn
        rows = []
        for p in picks:
            rows.append((
                window_start, window_end, session_type,
                p.get("sector", "未知"), p.get("code", ""),
                p.get("name", ""), p.get("asset_type", "stock"),
                p.get("score", 0), p.get("close_price", 0),
                p.get("rank", 0),
            ))
        if rows:
            c.executemany("""
                INSERT INTO sector_rotation_tracking
                (window_start, window_end, session_type, sector_name, code, name,
                 asset_type, score, close_price, pick_rank)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, rows)
            c.commit()
        logger.info(f"轮动追踪入库: {session_type} {window_start}~{window_end}, {len(rows)} 只")
        return len(rows)

    def upsert_pick_frequency(self, code: str, session_type: str) -> None:
        """更新股票累计命中次数。"""
        c = self.conn
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("""
            INSERT INTO pick_frequency (code, session_type, total_hits, last_hit_date, first_hit_date)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(code, session_type) DO UPDATE SET
                total_hits = total_hits + 1,
                last_hit_date = ?
        """, (code, session_type, today, today, today))
        c.commit()

    def get_rotation_window(self, sector: str = None, session_type: str = None,
                            limit: int = 50) -> pd.DataFrame:
        """查询轮动窗口内的选股记录。"""
        c = self.conn
        conds, params = [], []
        if sector:
            conds.append("sector_name = ?"); params.append(sector)
        if session_type:
            conds.append("session_type = ?"); params.append(session_type)
        where = " AND ".join(conds) if conds else "1=1"
        rows = c.execute(
            f"SELECT * FROM sector_rotation_tracking WHERE {where} ORDER BY window_start DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_pick_frequency(self, session_type: str = None, min_hits: int = 1) -> pd.DataFrame:
        """获取股票累计命中排行榜。"""
        c = self.conn
        conds = ["total_hits >= ?"]
        params = [min_hits]
        if session_type:
            conds.append("session_type = ?")
            params.append(session_type)
        where = " AND ".join(conds)
        rows = c.execute(
            f"SELECT * FROM pick_frequency WHERE {where} ORDER BY total_hits DESC",
            params
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_sector_hit_counts(self, window_start: str = None) -> dict:
        """统计各板块近两周的命中只数。"""
        c = self.conn
        if window_start is None:
            window_start = (datetime.now() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        rows = c.execute("""
            SELECT sector_name, COUNT(*) AS cnt
            FROM sector_rotation_tracking
            WHERE window_start >= ?
            GROUP BY sector_name
            ORDER BY cnt DESC
        """, (window_start,)).fetchall()
        return {r["sector_name"]: r["cnt"] for r in rows}

    def bulk_insert_prices(self, rows: list[tuple]) -> int:
        """批量写入归一化价格数据。rows: [(code, date, close, pct_chg, vol, amount), ...]"""
        if not rows:
            return 0
        c = self.conn
        c.executemany("""
            INSERT OR REPLACE INTO daily_price
            (code, date, close, pct_chg, vol, amount)
            VALUES (?,?,?,?,?,?)
        """, rows)
        c.commit()
        return len(rows)

    # ============================================
    #  持久化基本面表（v4 新增）
    # ============================================

    FUND_COLUMNS = [
        "code", "name", "industry", "area", "list_date", "market",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio",
        "total_mv", "circ_mv", "roe", "roa", "gross_margin", "debt_ratio",
        "revenue_growth", "profit_growth", "dedt_net_profit", "eps", "bps",
        "report_period", "valuation_date",
    ]

    def upsert_fundamentals(self, df: pd.DataFrame) -> int:
        """批量写入/更新基本面快照。df 需包含 FUND_COLUMNS 中的列。"""
        if df is None or len(df) == 0:
            return 0
        cols = self.FUND_COLUMNS
        rows = []
        for _, r in df.iterrows():
            row = []
            for c in cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    row.append(None)
                else:
                    row.append(v)
            rows.append(tuple(row))
        c = self.conn
        q = (f"INSERT OR REPLACE INTO fundamentals ({','.join(cols)}) "
             f"VALUES ({','.join('?' for _ in cols)})")
        c.executemany(q, rows)
        c.commit()
        return len(rows)

    def get_fundamentals_table(self, codes: list[str] = None) -> pd.DataFrame:
        """读取基本面表。codes 为空返回全表。"""
        c = self.conn
        if codes:
            placeholders = ",".join("?" for _ in codes)
            rows = c.execute(
                f"SELECT * FROM fundamentals WHERE code IN ({placeholders})", codes
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM fundamentals").fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ============================================
    #  PIT 基本面（时点正确，消除前视）
    # ============================================

    PIT_FIN_COLUMNS = [
        "code", "end_date", "ann_date", "roe", "roa", "gross_margin",
        "debt_ratio", "revenue_growth", "profit_growth", "eps", "eps_ttm",
        "bps", "report_type",
    ]

    PIT_VAL_COLUMNS = [
        "code", "trade_date", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
        "dv_ratio", "total_mv", "circ_mv",
    ]

    def existing_pit_fin_keys(self) -> set:
        """返回已入库的 (code, end_date) 集合, 供 DB-first 增量采集。"""
        rows = self.conn.execute(
            "SELECT code, end_date FROM fundamentals_pit"
        ).fetchall()
        return {(r["code"], r["end_date"]) for r in rows}

    def upsert_fundamentals_pit(self, df: pd.DataFrame) -> int:
        """批量写入/更新 PIT 财务报表（按 code+end_date 幂等）。"""
        if df is None or len(df) == 0:
            return 0
        cols = self.PIT_FIN_COLUMNS
        rows = []
        for _, r in df.iterrows():
            row = []
            for c in cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    row.append(None)
                else:
                    row.append(v)
            rows.append(tuple(row))
        c = self.conn
        q = (f"INSERT OR REPLACE INTO fundamentals_pit ({','.join(cols)}) "
             f"VALUES ({','.join('?' for _ in cols)})")
        c.executemany(q, rows)
        c.commit()
        return len(rows)

    def get_fundamentals_pit_all(self) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT * FROM fundamentals_pit"
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def existing_pit_val_dates(self) -> set:
        """返回已采集估值的交易日集合, 供 daily_basic_pit DB-first 增量。"""
        rows = self.conn.execute(
            "SELECT DISTINCT trade_date FROM daily_basic_pit"
        ).fetchall()
        return {r["trade_date"] for r in rows}

    def upsert_daily_basic_pit(self, df: pd.DataFrame) -> int:
        """批量写入/更新 PIT 估值（按 code+trade_date 幂等）。"""
        if df is None or len(df) == 0:
            return 0
        cols = self.PIT_VAL_COLUMNS
        rows = []
        for _, r in df.iterrows():
            row = []
            for c in cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    row.append(None)
                else:
                    row.append(v)
            rows.append(tuple(row))
        c = self.conn
        q = (f"INSERT OR REPLACE INTO daily_basic_pit ({','.join(cols)}) "
             f"VALUES ({','.join('?' for _ in cols)})")
        c.executemany(q, rows)
        c.commit()
        return len(rows)

    def get_daily_basic_pit_all(self) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT * FROM daily_basic_pit"
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_price_batch(self, codes: list[str], date_str: str) -> dict[str, float]:
        """批量查询指定日期多只股票的收盘价（用于T+N验证）。"""
        if not codes or not date_str:
            return {}
        c = self.conn
        placeholders = ",".join("?" for _ in codes)
        rows = c.execute(
            f"SELECT code, close FROM daily_price WHERE date=? AND code IN ({placeholders})",
            [date_str] + codes
        ).fetchall()
        return {r["code"]: r["close"] for r in rows}


# ---- 全局单例（双 DB：选股结果 + 行情数据） ----
_sel_db: Optional[StockDB] = None
_mkt_db: Optional[StockDB] = None

DB_SELECTIONS = "data_cache/selections.db"
DB_MARKET = "data_cache/market.db"


def get_db(db_path: str = None) -> StockDB:
    """获取选股结果 DB（默认 selections.db）。"""
    global _sel_db
    if db_path is None:
        db_path = DB_SELECTIONS
    if _sel_db is None:
        _sel_db = StockDB(db_path)
    return _sel_db


def get_market_db(db_path: str = None) -> StockDB:
    """获取行情数据 DB（默认 market.db）。"""
    global _mkt_db
    if db_path is None:
        db_path = DB_MARKET
    if _mkt_db is None:
        _mkt_db = StockDB(db_path)
    return _mkt_db
