"""
westock_cli.py - westock-data CLI 共享封装模块

从 multifactor.py / daily_brief.py / verify_picks.py 等文件中
抽取的公共 CLI 调用逻辑，消除重复代码。

用法:
    from westock_cli import WestockCLI
    cli = WestockCLI(cache_dir="data_cache")
    df = cli.get_kline("000001", days=120)
    df = cli.get_index_kline("000001", days=60)
    df = cli.get_stock_list()
    df = cli.get_fundamentals(["000001", "600519"])
"""

import subprocess
import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _kill_process_tree(pid: int) -> None:
    """终止进程树（跨平台）。原代码在各文件中重复，现统一在此。"""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10
            )
        else:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired) as e:
        logger.warning(f"终止进程 {pid} 时出错: {e}")


def run_westock(args: list[str], timeout: int = 30, max_retries: int = 3) -> str:
    """
    调用 westock-data CLI 并返回 stdout 输出。
    包含重试机制和超时处理。

    Args:
        args: CLI 参数列表，如 ["query", "--stock", "000001", "--type", "kline"]
        timeout: 超时秒数
        max_retries: 最大重试次数

    Returns:
        CLI 的 stdout 输出（字符串）

    Raises:
        RuntimeError: 所有重试均失败时抛出
    """
    cmd = ["westock-data"] + args
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"westock CLI 调用 (尝试 {attempt}/{max_retries}): {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning(f"westock CLI 超时 (尝试 {attempt}/{max_retries})")
            last_error = "timeout"
        except subprocess.CalledProcessError as e:
            logger.warning(f"westock CLI 返回非零退出码 {e.returncode} (尝试 {attempt}/{max_retries}): {e.stderr[:200]}")
            last_error = f"exit_code={e.returncode}"
        except FileNotFoundError:
            logger.error("westock-data CLI 未找到，请确认已安装 westock-data skill")
            raise RuntimeError("westock-data CLI 不可用")
        except Exception as e:
            logger.warning(f"westock CLI 异常 (尝试 {attempt}/{max_retries}): {type(e).__name__}: {e}")
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(1 * attempt)  # 递增等待

    raise RuntimeError(f"westock CLI 调用失败（重试 {max_retries} 次）: {last_error}")


class WestockCLI:
    """westock-data CLI 封装，提供缓存和便捷方法。"""

    def __init__(self, cache_dir: str = "data_cache", cache_expiry_hours: int = 12):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_seconds = cache_expiry_hours * 3600
        self._timeout = 30
        self._max_retries = 3

    # ---- 缓存管理 ----

    def _cache_path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_expiry_seconds:
            logger.debug(f"缓存过期: {key}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"读取缓存失败 {key}: {e}")
            return None

    def _write_cache(self, key: str, data: dict) -> None:
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        except OSError as e:
            logger.warning(f"写入缓存失败 {key}: {e}")

    # ---- 数据接口 ----

    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表。"""
        cache_key = "stock_list"
        cached = self._read_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        output = run_westock(["query", "--type", "stock_list", "--market", "A股"],
                             timeout=self._timeout, max_retries=self._max_retries)
        data = json.loads(output)
        self._write_cache(cache_key, data)
        return pd.DataFrame(data)

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取个股K线数据。
        code: 股票代码，如 "000001"
        days: 获取近多少个交易日
        adjust: 复权方式 qfq(前复权) / hfq(后复权) / none(不复权)
        """
        cache_key = f"kline_{code}_{days}_{adjust}"
        cached = self._read_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        output = run_westock(
            ["query", "--type", "kline", "--stock", code,
             "--days", str(days), "--adjust", adjust],
            timeout=self._timeout, max_retries=self._max_retries
        )
        data = json.loads(output)
        self._write_cache(cache_key, data)
        return pd.DataFrame(data)

    def get_index_kline(self, code: str, days: int = 60) -> pd.DataFrame:
        """获取指数K线数据。code: 指数代码，如 "000001"(上证指数)"""
        cache_key = f"index_kline_{code}_{days}"
        cached = self._read_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        output = run_westock(
            ["query", "--type", "kline", "--index", code, "--days", str(days)],
            timeout=self._timeout, max_retries=self._max_retries
        )
        data = json.loads(output)
        self._write_cache(cache_key, data)
        return pd.DataFrame(data)

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """获取股票基本面数据（PE/PB/ROE/毛利率/营收增速等）。"""
        cache_key = f"fundamentals_{'_'.join(codes[:10])}_{len(codes)}"
        cached = self._read_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        output = run_westock(
            ["query", "--type", "fundamentals", "--stocks", ",".join(codes)],
            timeout=self._timeout * 2,  # 基本面数据量大，超时加倍
            max_retries=self._max_retries
        )
        data = json.loads(output)
        self._write_cache(cache_key, data)
        return pd.DataFrame(data)

    def get_sector_mapping(self) -> dict[str, str]:
        """获取股票-板块映射。返回 {code: sector_name}"""
        cache_key = "sector_mapping"
        cached = self._read_cache(cache_key)
        if cached:
            return cached

        output = run_westock(
            ["query", "--type", "sector_mapping"],
            timeout=self._timeout, max_retries=self._max_retries
        )
        data = json.loads(output)
        self._write_cache(cache_key, data)
        return data


# ---- 全局单例 ----
_cli_instance: Optional[WestockCLI] = None


def get_cli() -> WestockCLI:
    """获取全局 WestockCLI 实例（单例模式）。"""
    global _cli_instance
    if _cli_instance is None:
        _cli_instance = WestockCLI()
    return _cli_instance


def sector_of(code: str, mapping: Optional[dict] = None) -> str:
    """
    查询股票所属板块。原代码在各文件中重复，现统一在此。
    mapping: 可选的预加载映射，避免重复查询
    """
    if mapping and code in mapping:
        return mapping[code]
    try:
        m = get_cli().get_sector_mapping()
        return m.get(code, "未知")
    except Exception as e:
        logger.warning(f"查询板块失败 {code}: {e}")
        return "未知"
