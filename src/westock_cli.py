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


from sys_config import get_encoding, get_npx_path
_ENCODING = get_encoding()
WESTOCK_CLI_CMD = [get_npx_path(), "-y", "westock-data-skillhub@1.0.5"]


def _to_ws_code(code: str) -> str:
    """6位代码 → westock sh/sz 前缀格式。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("8", "4")):
        return f"bj{code}"
    else:
        return f"sz{code}"


def _to_index_ws_code(code: str) -> str:
    """指数代码 → westock sh/sz 前缀格式。

    必须与个股区分：上证指数代码 000001 与平安银行(000001.SZ) 撞码，
    若走 _to_ws_code 会被转成 sz000001 返回平安银行个股行情，污染 L0 闸门。
    规则：上证系列(000xxx) → sh；深证/创业板系列(399xxx) → sz。
    """
    code = str(code).zfill(6)
    if code.startswith("000"):
        return f"sh{code}"
    return f"sz{code}"


def _parse_pipe_table(output: str) -> list[dict]:
    """解析 westock CLI 的 pipe-delimited 表格输出。"""
    lines = output.strip().split("\n")
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    rows = []
    for line in lines[2:]:  # skip header separator
        if not line.strip():
            continue
        vals = [v.strip() for v in line.split("|")[1:-1]]
        if len(vals) >= len(headers):
            rows.append(dict(zip(headers, vals)))
    return rows


_DATE_COLS = {"date", "time", "datetime", "timestamp", "trading_date"}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 westock pipe 表格解析出的字符串列转换为数值类型。

    westock 输出为纯文本表格，所有数值（close/open/high/volume 等）解析后
    都是字符串类型（pandas 3.x 下为 str/StringDtype）；若直接对字符串列做
    .mean()/.rolling() 会抛 "Cannot perform reduction 'mean' with string dtype"。

    策略：对每个非数值、非日期列尝试 pd.to_numeric(errors="coerce")，仅当该列
    **所有非空值**都能成功转为数值时才替换；否则视为日期/名称类文本，保留原样。
    注意 pandas 3.x 已移除 errors="ignore"。
    """
    for col in df.columns:
        if col.lower() in _DATE_COLS:
            continue
        if str(df[col].dtype).startswith(("int", "float")):
            continue
        nonnull = df[col].notna()
        if not nonnull.any():
            continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
        except (ValueError, TypeError):
            continue
        # 只有「全部非空值都可解析」才认定该列是数值列
        if converted.notna().sum() == int(nonnull.sum()):
            df[col] = converted
    return df


def run_westock(args: list[str], timeout: int = 30, max_retries: int = 3) -> str:
    """
    调用 westock-data CLI 并返回 stdout 输出。
    包含重试机制、超时处理、子进程清理。
    """
    cmd = WESTOCK_CLI_CMD + args
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
                encoding=_ENCODING,
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
            logger.debug("westock-data CLI 未安装，将使用 tushare/akshare 回退")
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
    """westock-data CLI 封装，失败时自动回退: tushare → akshare。"""

    def __init__(self, cache_dir: str = "data_cache", cache_expiry_hours: int = 12):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_seconds = cache_expiry_hours * 3600
        self._timeout = 30
        self._max_retries = 3
        self._akshare = None  # 懒加载
        self._tushare = None  # 懒加载

    @property
    def _ts(self):
        """懒加载 tushare provider（优先级高于 akshare）。"""
        if self._tushare is None:
            try:
                from tushare_provider import get_tushare
                self._tushare = get_tushare()
                logger.info("已启用 tushare 数据源作为后备")
            except ImportError as e:
                logger.info(f"tushare 未安装: {e}")
                self._tushare = None
            except Exception as e:
                logger.warning(f"tushare 初始化失败: {e}")
                self._tushare = None
        return self._tushare

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
            return json.loads(path.read_text(encoding=_ENCODING))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"读取缓存失败 {key}: {e}")
            return None

    def _write_cache(self, key: str, data: dict) -> None:
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding=_ENCODING)
        except OSError as e:
            logger.warning(f"写入缓存失败 {key}: {e}")

    @property
    def _ak(self):
        """懒加载 akshare provider。"""
        if self._akshare is None:
            try:
                from akshare_provider import get_akshare
                self._akshare = get_akshare()
                logger.info("已启用 akshare 数据源作为后备")
            except ImportError:
                logger.warning("akshare 未安装，仅使用 westock-data CLI")
                self._akshare = None
        return self._akshare

    # ---- 数据接口 ----

    def get_stock_list(self) -> pd.DataFrame:
        """获取全A股股票列表。优先 tushare，回退 akshare，最终回退 westock-data 指数成份股。"""
        if self._ts:
            try:
                return self._ts.get_stock_list()
            except Exception as e2:
                logger.info(f"tushare 股票列表失败: {e2}，回退 akshare")
        if self._ak:
            try:
                return self._ak.get_stock_list()
            except Exception as e3:
                logger.info(f"akshare 股票列表失败: {e3}，回退 westock-data 指数成份股")
        return self._get_stock_list_westock()

    def _get_stock_list_westock(self) -> pd.DataFrame:
        """westock-data 回退：通过指数成份股 + 批量 K 线构建股票列表。"""
        indices = ["sh000300", "sh000905"]  # 沪深300 + 中证500 ≈ 800 只
        all_codes = {}  # code → name

        for idx in indices:
            try:
                output = run_westock(
                    ["index", "constituent", idx],
                    timeout=30, max_retries=1
                )
                rows = _parse_pipe_table(output)
                for row in rows:
                    raw_code = row.get("code", "")
                    code = raw_code.replace("sh", "").replace("sz", "").replace("bj", "").zfill(6)
                    if code not in all_codes:
                        all_codes[code] = row.get("name", code)
            except Exception as e:
                logger.debug(f"westock-data 指数成份股 {idx} 失败: {e}")

        if not all_codes:
            logger.warning("westock-data 指数成份股为空，无法构建股票列表")
            return pd.DataFrame()

        # 批量获取 K 线（limit=2 拿今日+昨日，计算量比）
        codes_list = list(all_codes.keys())
        all_kline = []
        batch_size = 80  # CPU-safe: 小批次顺序执行
        for i in range(0, len(codes_list), batch_size):
            batch = codes_list[i:i + batch_size]
            ws_codes = ",".join(_to_ws_code(c) for c in batch)
            try:
                output = run_westock(
                    ["kline", ws_codes, "--period", "day", "--limit", "2"],
                    timeout=90, max_retries=2
                )
                lines = output.strip().split("\n")
                today_data = {}
                prev_vol_data = {}
                for line in lines:
                    if not line or "|" not in line or "[Batch]" in line or "symbol" in line:
                        continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 10:
                        continue
                    symbol = parts[1].replace("sh", "").replace("sz", "").replace("bj", "").zfill(6)
                    try:
                        close_p = float(parts[4])
                        high_p = float(parts[5])
                        low_p = float(parts[6])
                        volume = float(parts[7])
                        amount = float(parts[8])
                        exchange = float(parts[9])
                    except (ValueError, IndexError):
                        continue
                    if symbol not in today_data:
                        today_data[symbol] = {
                            "close": close_p, "high": high_p, "low": low_p,
                            "volume": volume, "amount": amount, "change_pct": exchange
                        }
                    elif symbol not in prev_vol_data:
                        prev_vol_data[symbol] = volume

                for code, data in today_data.items():
                    prev_vol = prev_vol_data.get(code, data["volume"])
                    vol_ratio = data["volume"] / prev_vol if prev_vol > 0 else 1.0
                    amp = ((data["high"] - data["low"]) / data["close"] * 100
                           if data["close"] > 0 else 0)
                    all_kline.append({
                        "code": code,
                        "close": data["close"],
                        "change_pct": data["change_pct"],
                        "volume": data["volume"],
                        "amount": data["amount"],
                        "volume_ratio": round(vol_ratio, 2),
                        "amplitude": round(amp, 2),
                    })
            except Exception as e:
                logger.debug(f"westock-data K 线批次 {i}-{i+batch_size} 失败: {e}")
            time.sleep(0.5)  # CPU-safe

        if not all_kline:
            logger.warning("westock-data K 线数据为空")
            return pd.DataFrame()

        kdf = pd.DataFrame(all_kline)
        named = pd.DataFrame([{"code": c, "name": n} for c, n in all_codes.items()])
        df = named.merge(kdf, on="code", how="left")
        logger.info(f"westock-data 指数成份股: {len(df)} 只 (K线覆盖 {kdf['change_pct'].notna().sum()})")
        return df

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> pd.DataFrame:
        """获取个股K线数据。优先 westock-data，失败回退 tushare → akshare。"""
        cache_key = f"kline_{code}_{days}_{adjust}"
        cached = self._read_cache(cache_key)
        if cached:
            df = pd.DataFrame(cached)
            if "last" in df.columns and "close" not in df.columns:
                df = df.rename(columns={"last": "close"})
            df = _coerce_numeric(df)
            return df

        try:
            ws_code = _to_ws_code(code)
            fq_map = {"qfq": "qfq", "hfq": "hfq", "": "bfq"}
            fq = fq_map.get(adjust, "qfq")
            output = run_westock(
                ["kline", ws_code, "--period", "day", "--limit", str(days), "--fq", fq],
                timeout=self._timeout, max_retries=1
            )
            rows = _parse_pipe_table(output)
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={"last": "close"})
                df = _coerce_numeric(df)
                self._write_cache(cache_key, df.to_dict(orient="records"))
                return df
        except Exception as e:
            logger.debug(f"westock-data K线失败 {code}: {e}，回退 tushare")
            if self._ts:
                try:
                    return self._ts.get_kline(code, days, adjust)
                except Exception as e2:
                    logger.debug(f"tushare K线失败 {code}: {e2}，回退 akshare")
            if self._ak:
                return self._ak.get_kline(code, days, adjust)
            return pd.DataFrame()

    def get_index_kline(self, code: str, days: int = 60) -> pd.DataFrame:
        """获取指数K线数据。优先 westock-data，失败回退 tushare → akshare。"""
        cache_key = f"index_kline_{code}_{days}"
        cached = self._read_cache(cache_key)
        if cached:
            df = pd.DataFrame(cached)
            if "last" in df.columns and "close" not in df.columns:
                df = df.rename(columns={"last": "close"})
            df = _coerce_numeric(df)
            return df

        try:
            ws_code = _to_index_ws_code(code)
            output = run_westock(
                ["kline", ws_code, "--period", "day", "--limit", str(days)],
                timeout=self._timeout, max_retries=1
            )
            rows = _parse_pipe_table(output)
            if rows:
                df = pd.DataFrame(rows)
                df = df.rename(columns={"last": "close"})  # westock → 系统标准列名
                df = _coerce_numeric(df)
                self._write_cache(cache_key, df.to_dict(orient="records"))
                return df
        except Exception as e:
            logger.debug(f"westock-data 指数K线失败 {code}: {e}，回退 tushare")
            if self._ts:
                try:
                    return self._ts.get_index_kline(code, days)
                except Exception as e2:
                    logger.debug(f"tushare 指数K线失败 {code}: {e2}，回退 akshare")
            if self._ak:
                return self._ak.get_index_kline(code, days)
            return pd.DataFrame()

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """获取基本面数据（profile）。优先 westock-data，失败回退 tushare。"""
        import hashlib
        key_hash = hashlib.md5(",".join(sorted([str(c) for c in codes[:30]])).encode()).hexdigest()[:12]
        cache_key = f"fundamentals_{key_hash}_{len(codes)}"
        cached = self._read_cache(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            # profile 支持批量，代码逗号分隔
            ws_codes = ",".join(_to_ws_code(c) for c in codes[:50])  # batch up to 50
            output = run_westock(
                ["profile", ws_codes],
                timeout=self._timeout, max_retries=1
            )
            rows = _parse_pipe_table(output)
            if rows:
                df = pd.DataFrame(rows)
                # 映射列名
                col_map = {"code": "code", "industry": "sector", "business": "business"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                # 提取6位代码
                df["code"] = df["code"].str.replace("sh", "").str.replace("sz", "").str.replace("bj", "").str.zfill(6)
                self._write_cache(cache_key, df.to_dict(orient="records"))
                return df
        except Exception as e:
            logger.info(f"westock-data 基本面失败: {e}，回退 tushare")
            if self._ts:
                try:
                    return self._ts.get_fundamentals(codes)
                except Exception as e2:
                    logger.info(f"tushare 基本面失败: {e2}，回退 akshare")
            if self._ak:
                return self._ak.get_fundamentals(codes)
            return pd.DataFrame()

    def get_realtime_quote(self, code: str) -> dict:
        """获取单只实时行情。优先 westock-data，失败回退 akshare(腾讯直连)。

        返回: {name, code, price, prev_close, open}；price 为最新价。
        """
        try:
            from westock_helpers import batch_close_prices
            cm = batch_close_prices([code])
            if cm and code in cm and cm[code] and cm[code] > 0:
                return {"name": "", "code": code, "price": cm[code],
                        "prev_close": None, "open": None}
        except Exception as e:
            logger.debug(f"westock 实时行情失败 {code}: {e}，回退 akshare")

        if self._ak:
            try:
                q = self._ak.get_realtime_quote(code)
                if q and q.get("price") is not None:
                    return q
            except Exception as e:
                logger.debug(f"akshare 实时行情失败 {code}: {e}")

        raise RuntimeError(f"实时行情获取失败: {code}（westock 与 akshare 均不可用）")

    def get_sector_mapping(self, codes: list[str] = None) -> dict[str, str]:
        """获取板块映射。优先 tushare，回退 akshare，最终回退 westock-data profile。"""
        if self._ts:
            try:
                return self._ts.get_sector_mapping()
            except Exception as e2:
                logger.info(f"tushare 板块映射失败: {e2}，回退 akshare")
        if self._ak:
            try:
                return self._ak.get_sector_mapping()
            except Exception as e3:
                logger.info(f"akshare 板块映射失败: {e3}，回退 westock-data profile")
        if codes:
            return self._get_sector_mapping_from_profile(codes)
        return {}

    def _get_sector_mapping_from_profile(self, codes: list[str]) -> dict[str, str]:
        """通过 westock-data profile 批量获取行业映射（max 30 只/批，CPU-safe）。"""
        if not codes:
            return {}
        mapping = {}
        batch_size = 30
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            ws_codes = ",".join(_to_ws_code(c) for c in batch)
            try:
                output = run_westock(
                    ["profile", ws_codes],
                    timeout=30, max_retries=1
                )
                rows = _parse_pipe_table(output)
                for row in rows:
                    raw_code = row.get("code", "")
                    code = raw_code.replace("sh", "").replace("sz", "").replace("bj", "").zfill(6)
                    industry = row.get("industry", "")
                    if industry:
                        mapping[code] = industry
            except Exception as e:
                logger.debug(f"westock-data profile 批次 {i} 失败: {e}")
            time.sleep(0.3)  # CPU-safe
        logger.info(f"westock-data profile 板块映射: {len(mapping)} 只")
        return mapping

    def get_sector_list(self) -> pd.DataFrame:
        """获取板块行情列表。westock-data 不直接支持 → 回退 tushare。"""
        if self._ts:
            try:
                return self._ts.get_sector_list()
            except Exception as e2:
                logger.info(f"tushare 板块列表失败: {e2}，回退 akshare")
        if self._ak:
            return self._ak.get_sector_list()
        return pd.DataFrame()


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
