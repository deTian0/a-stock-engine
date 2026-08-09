"""
import_local_data.py — 本地数据导入器 (CPU安全版)

将后复权 CSV + ETF CSV 数据导入 SQLite market_data_cache。
策略：单文件处理 / 1000行块写入 / 文件间休眠 / 纯串行。

后复权数据: 81个日文件，每个包含当日全A股截面数据
ETF数据: 2882个CSV文件，单个ETF历史日线

用法:
    python import_local_data.py
"""

import sys, os, time, logging
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from database import get_market_db

logger = logging.getLogger(__name__)

# === 配置 ===
HOUFUQUAN_DIR = r"D:\Download\BaiduNetdiskDownload\A股数据\后复权"   # 2020-2026 全部
ETF_DIR = r"D:\Download\BaiduNetdiskDownload\A股数据\ETF日线行情\行情数据"
STOCK_BASIC = r"D:\Download\BaiduNetdiskDownload\A股数据\A股日线\stock_basic.parquet"
PARQUET_PRICE_DIR = r"D:\Download\BaiduNetdiskDownload\A股数据\A股日线\1d_price"

CHUNK_SIZE = 1000          # 每块行数
FILE_SLEEP_SEC = 0.5       # 文件间休眠
CHUNK_SLEEP_SEC = 0.05     # 块间休眠
MAX_FILES = None            # None=全部, or e.g. 10 for test


def import_houfuquan(db, data_dir: str):
    """导入后复权日截面数据。"""
    csv_dir = Path(data_dir)
    files = sorted(csv_dir.rglob("*.csv"), key=lambda p: (p.parent.name, p.name))
    if MAX_FILES:
        files = files[:MAX_FILES]

    logger.info(f"后复权数据: {len(files)} 个文件")
    total_rows = 0

    for fi, fpath in enumerate(files):
        fname = fpath.stem
        date_str = fname.split("_")[0]  # "2026-01-05"
        logger.info(f"  [{fi+1}/{len(files)}] {date_str}...")

        try:
            # 只读需要的列（注意精确列名）
            usecols = [
                "日期", "代码", "名称", "所属行业",
                "收盘价", "滚动市盈率", "市净率",
                "总市值（元）", "流通市值（元）", "成交量（股）", "换手率", "涨幅%",
                "3日涨幅%", "6日涨幅%", "10日涨幅%", "25日涨幅%", "振幅%"
            ]
            df = pd.read_csv(fpath, encoding="utf-8", usecols=usecols, dtype={"代码": str})
            df["代码"] = df["代码"].str.zfill(6)
            df = df.rename(columns={
                "日期": "date", "代码": "code", "名称": "name",
                "所属行业": "sector", "收盘价": "close",
                "滚动市盈率": "pe", "市净率": "pb",
                "总市值（元）": "market_cap", "流通市值（元）": "float_cap",
                "成交量（股）": "volume", "换手率": "turnover",
                "涨幅%": "change_pct",
                "3日涨幅%": "chg_3d", "6日涨幅%": "chg_6d",
                "10日涨幅%": "chg_10d", "25日涨幅%": "chg_25d",
                "振幅%": "amplitude",
            })

            # 分块写入 SQLite
            cache_key = f"daily_snapshot_{date_str}"
            db.cache_put(cache_key, "daily_snapshot", df, "local_csv")

            rows = len(df)
            total_rows += rows
            logger.debug(f"    写入 {rows} 行")

        except Exception as e:
            logger.error(f"  {fname} 导入失败: {e}")
            continue

        # CPU 保护
        time.sleep(FILE_SLEEP_SEC)

    logger.info(f"后复权导入完成: {total_rows} 行")
    return total_rows


def import_etf(db, data_dir: str):
    """导入 ETF 日线数据（合并为一个合集）。"""
    etf_dir = Path(data_dir)
    files = list(etf_dir.glob("*.csv"))
    if MAX_FILES:
        files = files[:min(MAX_FILES * 5, len(files))]

    logger.info(f"ETF 数据: {len(files)} 个文件")
    all_etfs = []

    for fi, fpath in enumerate(files):
        try:
            code = fpath.stem.split(".")[0]  # "159001"
            df = pd.read_csv(fpath, dtype={"ts_code": str})
            if len(df) == 0:
                continue
            df["code"] = code
            df = df.rename(columns={
                "trade_date": "date", "open": "open", "close": "close",
                "high": "high", "low": "low", "vol": "volume",
                "amount": "amount", "pct_chg": "change_pct",
                "csname": "name",
            })
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
            all_etfs.append(df[["code", "date", "name", "close", "open", "high", "low",
                               "volume", "amount", "change_pct"]])

            if (fi + 1) % 50 == 0:
                logger.info(f"  ETF 进度: {fi+1}/{len(files)}")
                time.sleep(CHUNK_SLEEP_SEC)

        except Exception as e:
            logger.debug(f"  ETF {fpath.name} 失败: {e}")

        # CPU 保护
        if (fi + 1) % 10 == 0:
            time.sleep(FILE_SLEEP_SEC)

    if all_etfs:
        combined = pd.concat(all_etfs, ignore_index=True)
        db.cache_put("etf_daily_all", "etf_daily", combined, "local_csv")
        logger.info(f"ETF 导入完成: {len(combined)} 行, {len(combined['code'].unique())} 只")
        return len(combined)

    return 0


def import_stock_basic(db):
    """导入股票基础信息。"""
    try:
        df = pd.read_parquet(STOCK_BASIC)
        db.cache_put("stock_basic", "reference", df, "local_parquet")
        logger.info(f"stock_basic 导入: {len(df)} 行")
        return len(df)
    except Exception as e:
        logger.error(f"stock_basic 导入失败: {e}")
        return 0


def import_parquet_prices(db, base_dir: str):
    """导入 parquet 日线价格数据 → daily_price 表 (CPU安全)。"""
    import glob
    base = Path(base_dir)
    files = sorted(base.rglob("*.parquet"))
    if MAX_FILES:
        files = files[:MAX_FILES]

    logger.info(f"Parquet 价格数据: {len(files)} 个文件")
    total_rows = 0

    for fi, fpath in enumerate(files):
        try:
            df = pd.read_parquet(str(fpath), columns=["code", "close", "pct_chg", "vol", "amount"])
            if len(df) == 0:
                continue

            df["code"] = df["code"].astype(str).str.zfill(6)
            date_str = fpath.stem[:8]  # "20260105"
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            rows = [
                (str(r.code), date_str,
                 float(r.close) if pd.notna(r.close) else 0.0,
                 float(r.pct_chg) if pd.notna(r.pct_chg) else 0.0,
                 float(r.vol) if pd.notna(r.vol) else 0.0,
                 float(r.amount) if pd.notna(r.amount) else 0.0)
                for _, r in df.iterrows()
            ]

            if rows:
                db.bulk_insert_prices(rows)
                total_rows += len(rows)

            if (fi + 1) % 50 == 0:
                logger.info(f"  Parquet 进度: {fi+1}/{len(files)}")
                time.sleep(0.1)

        except Exception as e:
            logger.debug(f"  {fpath.name}: {e}")

        time.sleep(0.05)  # CPU 保护

    logger.info(f"Parquet 导入完成: {total_rows} 行")
    return total_rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("=" * 60)
    logger.info("本地数据导入器启动 (CPU 安全模式)")
    logger.info("=" * 60)

    db = get_market_db()

    try:
        t1 = import_stock_basic(db)
        time.sleep(FILE_SLEEP_SEC)

        t2 = import_houfuquan(db, HOUFUQUAN_DIR)
        time.sleep(FILE_SLEEP_SEC)

        t3 = import_etf(db, ETF_DIR)
        time.sleep(FILE_SLEEP_SEC)

        t4 = import_parquet_prices(db, PARQUET_PRICE_DIR)

        logger.info(f"\n导入完成: stock_basic={t1}, 后复权={t2}行, ETF={t3}行, Parquet={t4}行")

        # 验证
        total_mb = Path("data_cache/a-stock-engine.db").stat().st_size / (1024 * 1024)
        logger.info(f"DB 总大小: {total_mb:.0f}MB")

    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        db.close()
        logger.info("DB 连接已关闭")


if __name__ == "__main__":
    main()
