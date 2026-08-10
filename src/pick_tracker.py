"""
pick_tracker.py — 选股命中追踪系统

追踪每只被选中股票的表现，分三个统计维度:
  1. 盘前累计命中数 (pre_market cumulative) — 全局生命周期累计
  2. 盘后累计命中数 (post_market cumulative) — 全局生命周期累计
  3. 盘前周期内命中数 (pre_market cycle) — 10交易日内重复命中次数

规则:
  - 首次命中启动 10 交易日追踪周期
  - 周期内再次命中 → 延长追踪周期 + 周期命中数+1
  - 同一天同一 session 多次命中 → 只计 1 次
  - 周期结束后重新开始

用法:
    from pick_tracker import track_picks, get_tracking_summary
    track_picks(df_with_picks, session_type='pre_market')
    summary = get_tracking_summary()
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from database import get_db

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CYCLE_TRADING_DAYS = 10
CYCLE_CALENDAR_DAYS = 14  # 10个交易日 ≈ 14个自然日


def _init_tracking_schema(db):
    """初始化追踪相关表。"""
    c = db.conn
    # 版本检查
    c.execute("CREATE TABLE IF NOT EXISTS tracking_schema_version (version INTEGER)")
    row = c.execute("SELECT version FROM tracking_schema_version").fetchone()
    if row and row["version"] >= SCHEMA_VERSION:
        return

    c.executescript("""
    -- 每日命中明细
    CREATE TABLE IF NOT EXISTS pick_tracking (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        code          TEXT    NOT NULL,
        name          TEXT,
        session_type  TEXT    NOT NULL,
        pick_date     TEXT    NOT NULL,
        cycle_id      TEXT,           -- 周期标识(首次命中日期)
        cycle_start   TEXT,           -- 周期开始日
        cycle_end     TEXT,           -- 周期预计结束日
        cycle_hits    INTEGER DEFAULT 1, -- 周期内累计命中数(实时)
        final_cycle_hits INTEGER DEFAULT 0, -- 周期结束时命中数
        cumulative    INTEGER DEFAULT 1,   -- 本session累计命中数
        is_cycle_start INTEGER DEFAULT 0,  -- 是否为周期首日
        created_at    TEXT    DEFAULT (datetime('now','localtime'))
    );

    -- 汇总统计(快速查询)
    CREATE TABLE IF NOT EXISTS pick_summary (
        code              TEXT NOT NULL,
        name              TEXT,
        session_type      TEXT NOT NULL,
        cumulative_hits   INTEGER DEFAULT 0,
        active_cycle_id   TEXT,
        active_cycle_hits INTEGER DEFAULT 0,
        active_cycle_start TEXT,
        active_cycle_end   TEXT,
        total_cycles      INTEGER DEFAULT 0,
        last_pick_date    TEXT,
        first_pick_date   TEXT,
        updated_at        TEXT DEFAULT (datetime('now','localtime')),
        PRIMARY KEY (code, session_type)
    );

    CREATE INDEX IF NOT EXISTS idx_pick_trk_date    ON pick_tracking(pick_date);
    CREATE INDEX IF NOT EXISTS idx_pick_trk_code    ON pick_tracking(code, session_type);
    CREATE INDEX IF NOT EXISTS idx_pick_trk_cycle   ON pick_tracking(cycle_id);
    CREATE INDEX IF NOT EXISTS idx_pick_sum_session ON pick_summary(session_type, cumulative_hits DESC);
    """)

    c.execute("INSERT OR REPLACE INTO tracking_schema_version (version) VALUES (?)",
              (SCHEMA_VERSION,))
    c.commit()
    logger.info("命中追踪表初始化完成")


class PickTracker:
    """选股命中追踪器。"""

    def __init__(self):
        self.db = get_db()
        _init_tracking_schema(self.db)

    def track_picks(self, df: pd.DataFrame, session_type: str,
                    pick_date: str = None) -> int:
        """
        批量记录选股命中。
        
        Args:
            df: 含 code, name 列的 DataFrame
            session_type: 'pre_market' 或 'post_market'
            pick_date: 命中日期，默认今天
        
        Returns:
            本次新记录的命中数
        """
        if df is None or len(df) == 0:
            return 0

        if pick_date is None:
            pick_date = datetime.now().strftime("%Y-%m-%d")

        codes = df["code"].tolist() if "code" in df.columns else []
        names = df.get("name", pd.Series([""] * len(df))).tolist() if "name" in df.columns else codes

        new_count = 0
        for code, name in zip(codes, names):
            code = str(code).zfill(6)
            try:
                if self._track_one(code, name, session_type, pick_date):
                    new_count += 1
            except Exception as e:
                logger.debug(f"追踪记录失败 {code}: {e}")

        self.db.commit()
        logger.info(f"命中追踪完成: {session_type} {pick_date}, 新增 {new_count} 条")
        return new_count

    def _track_one(self, code: str, name: str, session_type: str,
                   pick_date: str) -> bool:
        """记录单只股票的一次命中。返回是否新增。"""
        c = self.db.conn

        # 检查同一天同一session是否已记录
        exists = c.execute(
            "SELECT id FROM pick_tracking WHERE code=? AND session_type=? AND pick_date=?",
            (code, session_type, pick_date)
        ).fetchone()
        if exists:
            return False  # 同日不重复计数

        # 获取或创建汇总
        summary = c.execute(
            "SELECT * FROM pick_summary WHERE code=? AND session_type=?",
            (code, session_type)
        ).fetchone()

        if summary is None:
            # 首次命中: 创建汇总 + 新周期
            cycle_id = pick_date
            cycle_start = pick_date
            cycle_end = (datetime.strptime(pick_date, "%Y-%m-%d") +
                        timedelta(days=CYCLE_CALENDAR_DAYS)).strftime("%Y-%m-%d")
            cycle_hits = 1
            cumulative = 1
            is_cycle_start = 1

            c.execute("""
                INSERT INTO pick_summary
                (code, name, session_type, cumulative_hits,
                 active_cycle_id, active_cycle_hits, active_cycle_start, active_cycle_end,
                 total_cycles, last_pick_date, first_pick_date)
                VALUES (?,?,?,1, ?,1,?,?, 1,?,?)
            """, (code, name, session_type,
                  cycle_id, cycle_start, cycle_end,
                  pick_date, pick_date))
        else:
            # 已有记录: 更新
            cumulative = summary["cumulative_hits"] + 1

            # 检查是否在活动周期内
            active_cycle_id = summary["active_cycle_id"]
            active_cycle_end = summary["active_cycle_end"]

            if active_cycle_id and active_cycle_end and pick_date <= active_cycle_end:
                # 在周期内: 延长周期 + 增加命中
                cycle_id = active_cycle_id
                cycle_start = summary["active_cycle_start"]
                # 延长: 从今天起 +14天
                cycle_end = (datetime.strptime(pick_date, "%Y-%m-%d") +
                            timedelta(days=CYCLE_CALENDAR_DAYS)).strftime("%Y-%m-%d")
                cycle_hits = summary["active_cycle_hits"] + 1
                is_cycle_start = 0
            else:
                # 旧周期已结束: 开始新周期
                cycle_id = pick_date
                cycle_start = pick_date
                cycle_end = (datetime.strptime(pick_date, "%Y-%m-%d") +
                            timedelta(days=CYCLE_CALENDAR_DAYS)).strftime("%Y-%m-%d")
                cycle_hits = 1
                is_cycle_start = 1

            total_cycles = summary["total_cycles"] + (1 if is_cycle_start else 0)

            c.execute("""
                UPDATE pick_summary SET
                    name=?, cumulative_hits=?, active_cycle_id=?,
                    active_cycle_hits=?, active_cycle_start=?, active_cycle_end=?,
                    total_cycles=?, last_pick_date=?
                WHERE code=? AND session_type=?
            """, (name, cumulative, cycle_id,
                  cycle_hits, cycle_start, cycle_end,
                  total_cycles, pick_date,
                  code, session_type))

        # 插入明细
        c.execute("""
            INSERT INTO pick_tracking
            (code, name, session_type, pick_date, cycle_id,
             cycle_start, cycle_end, cycle_hits, cumulative, is_cycle_start)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (code, name, session_type, pick_date, cycle_id,
              cycle_start, cycle_end, cycle_hits, cumulative, is_cycle_start))

        return True

    def get_summary(self, session_type: str = None) -> pd.DataFrame:
        """获取命中汇总。"""
        c = self.db.conn
        if session_type:
            rows = c.execute(
                "SELECT * FROM pick_summary WHERE session_type=? "
                "ORDER BY cumulative_hits DESC",
                (session_type,)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM pick_summary ORDER BY session_type, cumulative_hits DESC"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_active_cycles(self, session_type: str = "pre_market") -> pd.DataFrame:
        """获取当前活动周期内的股票。"""
        c = self.db.conn
        today = datetime.now().strftime("%Y-%m-%d")
        rows = c.execute(
            "SELECT * FROM pick_summary WHERE session_type=? "
            "AND active_cycle_end >= ? "
            "ORDER BY active_cycle_hits DESC",
            (session_type, today)
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_tracking_report(self) -> str:
        """生成命中追踪报告(Markdown)。"""
        c = self.db.conn

        lines = ["## 选股命中追踪\n"]

        for stype, label in [("pre_market", "盘前"), ("post_market", "盘后")]:
            # 累计 Top 10
            rows = c.execute(
                "SELECT * FROM pick_summary WHERE session_type=? "
                "ORDER BY cumulative_hits DESC LIMIT 10",
                (stype,)
            ).fetchall()

            if rows:
                lines.append(f"\n### {label}累计命中 Top 10\n")
                lines.append("| 代码 | 名称 | 累计命中 | 周期数 | 当前周期内 | 周期状态 | 最近命中 |")
                lines.append("|------|------|----------|--------|-----------|----------|----------|")
                today = datetime.now().strftime("%Y-%m-%d")
                for r in rows:
                    code = r["code"]
                    name = r["name"] or code
                    cum = r["cumulative_hits"]
                    cycles = r["total_cycles"]
                    cycle_hits = r["active_cycle_hits"]
                    cycle_end = r["active_cycle_end"] or ""
                    active = "🟢 追踪中" if cycle_end and cycle_end >= today else "⚪ 已结束"
                    last = r["last_pick_date"] or "-"
                    lines.append(
                        f"| {code} | {name} | {cum} | {cycles} | {cycle_hits} | {active} | {last} |"
                    )

            # 活跃周期内高频命中
            active = c.execute(
                "SELECT * FROM pick_summary WHERE session_type=? "
                "AND active_cycle_end >= ? AND active_cycle_hits >= 2 "
                "ORDER BY active_cycle_hits DESC LIMIT 10",
                (stype, today)
            ).fetchall()

            if active:
                lines.append(f"\n### {label}当前周期高频命中\n")
                lines.append("| 代码 | 名称 | 周期内命中 | 累计 | 周期开始 | 周期截止 |")
                lines.append("|------|------|-----------|------|----------|----------|")
                for r in active:
                    lines.append(
                        f"| {r['code']} | {r['name'] or r['code']} | {r['active_cycle_hits']} | "
                        f"{r['cumulative_hits']} | {r['active_cycle_start']} | {r['active_cycle_end']} |"
                    )

        # 全日汇总
        pre = c.execute(
            "SELECT COUNT(DISTINCT code) as cnt FROM pick_summary WHERE session_type='pre_market'"
        ).fetchone()
        post = c.execute(
            "SELECT COUNT(DISTINCT code) as cnt FROM pick_summary WHERE session_type='post_market'"
        ).fetchone()
        pre_cum = c.execute(
            "SELECT SUM(cumulative_hits) as total FROM pick_summary WHERE session_type='pre_market'"
        ).fetchone()
        post_cum = c.execute(
            "SELECT SUM(cumulative_hits) as total FROM pick_summary WHERE session_type='post_market'"
        ).fetchone()

        lines.append(f"\n---\n")
        lines.append(f"**总计**: 盘前 {pre['cnt']} 只({pre_cum['total'] or 0} 次) | "
                     f"盘后 {post['cnt']} 只({post_cum['total'] or 0} 次)\n")

        return "\n".join(lines)


# ================================================================
#  便捷函数
# ================================================================

_tracker: Optional[PickTracker] = None


def get_tracker() -> PickTracker:
    global _tracker
    if _tracker is None:
        _tracker = PickTracker()
    return _tracker


def track_picks(df: pd.DataFrame, session_type: str,
                pick_date: str = None) -> int:
    """便捷函数: 记录选股命中。"""
    return get_tracker().track_picks(df, session_type, pick_date)


def get_tracking_summary(session_type: str = None) -> pd.DataFrame:
    return get_tracker().get_summary(session_type)


def get_tracking_report() -> str:
    return get_tracker().get_tracking_report()
