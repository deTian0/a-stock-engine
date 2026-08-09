"""
westock_cli.py - westock-data CLI 共享封装模块

从 multifactor.py / daily_brief.py / verify_picks.py 等文件中
抽取的公共 CLI 调用逻辑，消除重复代码。

用法:
    from data.westock import WestockCLI
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
import hashlib
import logging
import time
import signal
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
    包含重试机制、超时处理、子进程清理。

    Args:
        args: CLI 参数列表
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
        proc = None
        try:
            logger.debug(f"westock CLI 调用 (尝试 {attempt}/{max_retries}): {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(timeout=timeout)

            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, output=stdout, stderr=stderr
                )

            return stdout

        except subprocess.TimeoutExpired:
            logger.warning(f"westock CLI 超时 (尝试 {attempt}/{max_retries})")
            last_error = "timeout"
            # 必须清理子进程，否则会成为僵尸
            if proc is not None:
                try:
                    _kill_process_tree(proc.pid)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"子进程 {proc.pid} 无法正常终止，强制结束")
                    try:
                        proc.kill()
                        proc.wait(timeout=3)
                    except Exception:
                        pass

        except subprocess.CalledProcessError as e:
            logger.warning(
                f"westock CLI 返回非零退出码 {e.returncode} "
                f"(尝试 {attempt}/{max_retries}): {(e.stderr or '')[:200]}"
            )
            last_error = f"exit_code={e.returncode}"

        except FileNotFoundError:
            logger.error("westock-data CLI 未找到，请确认已安装 westock-data skill")
            raise RuntimeError("westock-data CLI 不可用")

        except Exception as e:
            logger.warning(
                f"westock CLI 异常 (尝试 {attempt}/{max_retries}): "
                f"{type(e).__name__}: {e}"
            )
            last_error = str(e)
            if proc is not None:
                try:
                    _kill_process_tree(proc.pid)
                except Exception:
                    pass

        if attempt < max_retries:
            wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
            logger.debug(f"  等待 {wait}s 后重试...")
            time.sleep(wait)

    raise RuntimeError(f"westock CLI 调用失败（重试 {max_retries} 次）: {last_error}")


class WestockCLI:
    """数据源封装，按 config.yaml data.source 选择数据源。"""

    def __init__(self, cache_dir: str = "data_cache", cache_expiry_hours: int = 12):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_seconds = cache_expiry_hours * 3600
        self._timeout = 30
        self._max_retries = 3
        self._akshare = None
        self._sina = None
        self._config_source = self._load_config_source()

    def _load_config_source(self) -> str:
        try:
            import yaml
            config_path = Path(__file__).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                return cfg.get("data", {}).get("source", "sina")
        except Exception:
            pass
        return "sina"

    @property
    def _primary(self):
        """按配置选择主数据源。"""
        src = self._config_source
        if src == "sina":
            if self._sina is None:
                try:
                    from data.sina import get_sina
                    self._sina = get_sina()
                    logger.info("数据源: 新浪 API")
                except ImportError:
                    logger.warning("sina_provider 不可用，回退 akshare")
                    return self._ak
            return self._sina
        elif src == "akshare":
            return self._ak
        # westock / default → sina
        return self._ak  # 实际回退链在各方法中

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

    @property
    def _ak(self):
        """懒加载 akshare provider。"""
        if self._akshare is None:
            try:
                from data.akshare import get_akshare
                self._akshare = get_akshare()
                logger.info("已启用 akshare 数据源作为后备")
            except ImportError:
                logger.warning("akshare 未安装，仅使用 westock-data CLI")
                self._akshare = None
        return self._akshare

    # ---- 数据接口 ----

    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表。"""
        try:
            df = self._primary.get_stock_list()
            if len(df) > 0:
                return df
        except Exception as e:
            logger.debug(f"主数据源 stock_list 失败: {e}")
        return pd.DataFrame()

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> pd.DataFrame:
        """个股K线。"""
        try:
            return self._primary.get_kline(code, days, adjust)
        except Exception:
            if self._ak:
                return self._ak.get_kline(code, days, adjust)
            return pd.DataFrame()

    def get_index_kline(self, code: str, days: int = 60) -> pd.DataFrame:
        """指数K线。"""
        try:
            return self._primary.get_index_kline(code, days)
        except Exception:
            if self._ak:
                return self._ak.get_index_kline(code, days)
            return pd.DataFrame()

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """基本面数据。"""
        try:
            return self._primary.get_fundamentals(codes)
        except Exception:
            if self._ak:
                return self._ak.get_fundamentals(codes)
            return pd.DataFrame()

    def get_sector_mapping(self) -> dict[str, str]:
        """板块映射。"""
        try:
            return self._primary.get_sector_mapping()
        except Exception:
            if self._ak:
                return self._ak.get_sector_mapping()
            return {}

    def get_sector_list(self) -> pd.DataFrame:
        """板块行情。"""
        try:
            return self._primary.get_sector_list()
        except Exception:
            if self._ak:
                return self._ak.get_sector_list()
            return pd.DataFrame()
        if cached:
            return cached

        try:
            output = run_westock(
                ["query", "--type", "sector_mapping"],
                timeout=self._timeout, max_retries=1
            )
            data = json.loads(output)
            self._write_cache(cache_key, data)
            return data
        except Exception as e:
            logger.info(f"westock-data 板块映射失败: {e}，回退 akshare")
            if self._ak:
                return self._ak.get_sector_mapping()
            return {}


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
