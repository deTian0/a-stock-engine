"""
选股业务逻辑层
"""
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from data.db import get_market_db
from engine.factors import score_stocks, filter_candidates, pick_top_by_sector

logger = logging.getLogger("selector.service")


class SelectorService:
    """选股服务 — 封装数据库访问 + 评分逻辑。"""

    def __init__(self):
        self.db = get_market_db()  # 使用默认路径 data_cache/market.db
        self.raw_conn = sqlite3.connect(str(self.db.db_path))

    def get_latest_date(self) -> str:
        row = self.raw_conn.execute("SELECT MAX(date) FROM daily_price").fetchone()
        return row[0] if row[0] else ""

    def load_snapshot(self, date_str: str) -> pd.DataFrame:
        rows = self.raw_conn.execute(
            "SELECT code, close, pct_chg, vol, amount FROM daily_price WHERE date=?",
            (date_str,)
        ).fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["code_raw", "close", "pct_chg", "vol", "amount"])
        df["code"] = df["code_raw"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
        df["sector"] = "其他"
        df["name"] = df["code"]

        try:
            fund_df = self.db.cache_get(f"daily_snapshot_{date_str}")
            if fund_df is not None and len(fund_df) > 0:
                fund_df["code"] = fund_df["code"].astype(str).str.zfill(6)
                for col in ["name", "sector", "pe", "pb", "market_cap",
                            "turnover", "amplitude",
                            "chg_3d", "chg_6d", "chg_10d", "chg_25d", "change_pct"]:
                    if col in fund_df.columns:
                        fund_map = fund_df.set_index("code")[col].to_dict()
                        df[col] = df["code"].map(fund_map)
        except Exception:
            pass

        return df

    def compute_regime(self, date_str: str) -> dict:
        r = self.raw_conn.execute(
            "SELECT AVG(close) FROM daily_price WHERE date=?", (date_str,)
        ).fetchone()
        current_avg = r[0] or 0

        r2 = self.raw_conn.execute(
            """SELECT AVG(daily_avg) FROM (
                SELECT date, AVG(close) as daily_avg FROM daily_price 
                WHERE date <= ? GROUP BY date ORDER BY date DESC LIMIT 60
            )""", (date_str,)
        ).fetchone()
        ma60 = r2[0] or current_avg

        return {
            "regime": "bull" if current_avg > ma60 else "bear",
            "avg": round(current_avg, 2),
            "ma60": round(ma60, 2),
        }

    def run(self, top_n: int = 20, max_per_sector: int = 5) -> dict:
        date_str = self.get_latest_date()
        market = self.compute_regime(date_str)

        df = self.load_snapshot(date_str)
        if len(df) == 0:
            return {"date": date_str, "regime": market, "picks": [], "stats": {"error": "无数据"}}

        df = filter_candidates(df)
        df = score_stocks(df)
        all_picks = pick_top_by_sector(df, max_per_sector)

        if market["regime"] == "bear":
            picks = [p for p in all_picks if p.get("composite_score", 0) > 50][:max(top_n // 2, 5)]
        else:
            picks = all_picks[:top_n]

        score_min = df["composite_score"].min()
        score_range = max(df["composite_score"].max() - score_min, 1)
        for p in picks:
            s = p.get("composite_score", 0)
            p["score_norm"] = round((s - score_min) / score_range * 100, 1)
            p["target"] = round(p["close"] * 1.05, 2)
            p["stop"] = round(p["close"] * 0.95, 2)

        # 板块分布统计
        sectors = {}
        for p in picks:
            s = p.get("sector", "其他")
            sectors[s] = sectors.get(s, 0) + 1

        return {
            "date": date_str,
            "regime": market,
            "picks": picks,
            "stats": {
                "universe": len(df),
                "candidates": len(all_picks),
                "selected": len(picks),
                "sectors": sectors,
            },
        }

    def close(self):
        if self.db:
            self.db.close()
        if self.raw_conn:
            self.raw_conn.close()
