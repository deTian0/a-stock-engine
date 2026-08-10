"""
health_check.py — 启动前健康检查（借鉴 vnpy engine registry 模式）

检查所有数据源是否可用，输出状态报告。任一关键源不可用 → 警告但继续。
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS = {}


def check_all() -> dict:
    """运行所有检查，返回 {source: status}。"""
    global RESULTS
    RESULTS = {}

    _check("tushare", _check_tushare)
    _check("westock_cli", _check_westock)
    _check("akshare", _check_akshare)
    _check("db_selections", _check_db_selections)
    _check("db_market", _check_db_market)

    ok = sum(1 for v in RESULTS.values() if v == "OK")
    warn = sum(1 for v in RESULTS.values() if v == "WARN")
    fail = sum(1 for v in RESULTS.values() if v == "FAIL")

    logger.info(f"健康检查: {ok} OK, {warn} WARN, {fail} FAIL")
    return RESULTS


def _check(name: str, fn):
    try:
        fn()
        RESULTS[name] = "OK"
    except Exception as e:
        msg = str(e)[:80]
        if "not found" in msg.lower() or "unavailable" in msg.lower():
            RESULTS[name] = "FAIL"
            logger.warning(f"  {name}: FAIL — {msg}")
        else:
            RESULTS[name] = "WARN"
            logger.warning(f"  {name}: WARN — {msg}")


def _check_tushare():
    from tushare_provider import get_tushare, TUSHARE_TOKEN
    if not TUSHARE_TOKEN:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    ts = get_tushare()
    df = ts.pro.stock_basic(exchange="", list_status="L", limit=5)
    if df is None or len(df) == 0:
        raise RuntimeError("stock_basic 返回空")
    logger.info(f"  tushare OK: {len(df)} stocks returned")


def _check_westock():
    from westock_helpers import batch_close_prices
    prices = batch_close_prices(["600519"])
    if not prices:
        raise RuntimeError("westock kline 返回空 (npx可能不在PATH)")
    logger.info(f"  westock OK: 600519={prices.get('600519')}")


def _check_akshare():
    try:
        import akshare
        logger.info(f"  akshare OK: v{akshare.__version__}")
    except ImportError:
        raise RuntimeError("akshare 未安装")


def _check_db_selections():
    from database import get_db
    db = get_db()
    cnt = db.conn.execute("SELECT COUNT(*) FROM stock_picks").fetchone()[0]
    logger.info(f"  selections.db OK: {cnt} picks")


def _check_db_market():
    from database import get_market_db
    mdb = get_market_db()
    cnt = mdb.conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0]
    logger.info(f"  market.db OK: {cnt} price rows")


def print_report():
    """控制台输出健康检查报告。"""
    results = check_all()
    print("\n" + "=" * 50)
    print("  数据源健康检查")
    print("=" * 50)
    for name, status in results.items():
        icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "?")
        print(f"  {icon} {name}: {status}")
    print("=" * 50)
    ok = sum(1 for v in results.values() if v == "OK")
    print(f"  总计: {ok}/{len(results)} 正常\n")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print_report()
