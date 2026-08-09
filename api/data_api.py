"""
数据查询 API
"""
from pathlib import Path
from flask import Blueprint, jsonify
from data.db import get_market_db

data_bp = Blueprint("data", __name__)


@data_bp.route("/summary")
def data_summary():
    """数据库概况。"""
    db = get_market_db()
    c = db.conn
    try:
        price = c.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_price").fetchone()
        snap = c.execute(
            "SELECT COUNT(*) FROM market_data_cache WHERE data_type='daily_snapshot'"
        ).fetchone()

        return jsonify({
            "daily_price_rows": price[0],
            "daily_price_range": f"{price[1]} ~ {price[2]}" if price[1] else "无数据",
            "fundamental_days": snap[0],
            "status": "ok",
        })
    finally:
        db.close()


@data_bp.route("/dates")
def available_dates():
    """所有可用交易日。"""
    db = get_market_db()
    c = db.conn
    try:
        rows = c.execute(
            "SELECT DISTINCT date FROM daily_price WHERE date >= '2020-01-01' ORDER BY date"
        ).fetchall()
        return jsonify({
            "dates": [r[0] for r in rows],
            "count": len(rows),
        })
    finally:
        db.close()


@data_bp.route("/market-regime")
def market_regime():
    """当前市场环境。"""
    db = get_market_db()
    c = db.conn
    try:
        r = c.execute("SELECT MAX(date) FROM daily_price").fetchone()
        latest = r[0]

        avg = c.execute("SELECT AVG(close) FROM daily_price WHERE date=?", (latest,)).fetchone()
        ma60 = c.execute(
            """SELECT AVG(daily_avg) FROM (
                SELECT date, AVG(close) as daily_avg FROM daily_price 
                WHERE date <= ? GROUP BY date ORDER BY date DESC LIMIT 60
            )""", (latest,)
        ).fetchone()

        current_avg = avg[0] or 0
        ma60_value = ma60[0] or current_avg

        return jsonify({
            "date": latest,
            "regime": "bull" if current_avg > ma60_value else "bear",
            "market_avg": round(current_avg, 2),
            "ma60": round(ma60_value, 2),
        })
    finally:
        db.close()
