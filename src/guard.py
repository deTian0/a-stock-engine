"""
guard.py - 全局保护性基础设施

提供所有入口模块共享的：
  - 信号处理（SIGINT/SIGTERM 优雅退出）
  - KeyboardInterrupt 包装
  - DB 连接清理
  - 日志轮转配置

用法:
    from guard import setup_protection, teardown_protection

    def main():
        setup_protection()
        try:
            # ... 业务逻辑 ...
        finally:
            teardown_protection()
"""

import os
import sys
import signal
import logging
import atexit
from pathlib import Path
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logging.getLogger().warning(f"收到信号 {name}，正在安全退出...")


def is_shutdown_requested() -> bool:
    """检查是否收到退出信号。"""
    return _shutdown_requested


def setup_protection():
    """注册信号处理器。非主线程调用会被忽略。"""
    global _shutdown_requested
    _shutdown_requested = False

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError):
        pass


def teardown_protection():
    """清理资源：关闭 DB 连接。"""
    try:
        from database import get_db
        db = get_db()
        db.close()
        logger.debug("DB 连接已关闭")
    except Exception:
        pass


def setup_logging(log_file: str = "data_cache/engine.log",
                  level: str = "INFO",
                  console: bool = True,
                  max_bytes: int = 10 * 1024 * 1024,
                  backup_count: int = 3):
    """
    配置日志（带轮转）。
    max_bytes: 单文件最大 10MB
    backup_count: 保留 3 个历史文件
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    ]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    logger.debug(f"日志已配置: {log_path} (max {max_bytes//(1024*1024)}MB × {backup_count})")


def check_disk_space(data_dir: str = "data_cache", max_mb: int = 1024):
    """
    检查缓存目录大小，超过阈值时告警并清理过期文件。
    max_mb: 默认 1GB
    """
    path = Path(data_dir)
    if not path.exists():
        return

    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    total_mb = total / (1024 * 1024)

    if total_mb > max_mb:
        logger.warning(f"缓存目录过大: {total_mb:.0f}MB (阈值 {max_mb}MB)，正在清理...")

        # 清理 DB 过期缓存
        try:
            from database import get_db
            db = get_db()
            db.cache_clear_expired()
        except Exception as e:
            logger.warning(f"清理 DB 缓存失败: {e}")

        # 清理旧的 JSON 缓存文件（超过 24 小时）
        import time
        cutoff = time.time() - 24 * 3600
        deleted = 0
        for f in path.rglob("*.json"):
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except OSError:
                    pass

        if deleted > 0:
            logger.info(f"清理了 {deleted} 个过期缓存文件")

        # 重新计算
        new_total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        logger.info(f"清理后缓存大小: {new_total/(1024*1024):.0f}MB")
    else:
        logger.debug(f"缓存目录大小正常: {total_mb:.0f}MB")


def enforce_wal_checkpoint(db_path: str = "data_cache/a-stock-engine.db",
                           wal_threshold_mb: int = 50):
    """
    如果 WAL 文件超过阈值，执行 checkpoint 压缩。
    """
    wal_path = Path(db_path + "-wal")
    if not wal_path.exists():
        return

    wal_size = wal_path.stat().st_size / (1024 * 1024)
    if wal_size > wal_threshold_mb:
        logger.warning(f"WAL 文件过大: {wal_size:.0f}MB，执行 checkpoint...")
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            logger.info("WAL checkpoint 完成")
        except Exception as e:
            logger.warning(f"WAL checkpoint 失败: {e}")
