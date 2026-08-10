"""
westock_helpers.py — westock-data CLI 辅助函数（共享模块）

使用绝对路径 npx，避免 PATH 缺失问题。
"""

import subprocess
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# WorkBuddy 管理的 Node.js 路径
_NODE_DIR = Path.home() / ".workbuddy" / "binaries" / "node" / "versions" / "22.22.2"
if (_NODE_DIR / "npx.cmd").exists():
    _NPX = str(_NODE_DIR / "npx.cmd")
elif (_NODE_DIR / "npx").exists():
    _NPX = str(_NODE_DIR / "npx")
else:
    _NPX = "npx"


def _to_ws(code: str) -> str:
    """6位码 → sh/sz前缀。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("0", "2", "3")):
        return f"sz{code}"
    return f"bj{code}"


def batch_kline(codes: list[str], limit: int = 2) -> dict[str, list[float]]:
    """
    批量获取收盘价序列。返回 {code: [close_prices...]} (最新在前)。
    内部分批100只/次，失败记录warning。
    """
    result = {}
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        ws_codes = ",".join(_to_ws(c) for c in batch)
        try:
            r = subprocess.run(
                [_NPX, "-y", "westock-data-skillhub@1.0.5",
                 "kline", ws_codes, "--period", "day", "--limit", str(limit)],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode != 0:
                logger.warning(f"westock kline 返回码 {r.returncode}: {r.stderr[:120]}")
                continue
            if not r.stdout:
                continue

            for line in r.stdout.strip().split("\n"):
                if "|" not in line or "symbol" in line or "Batch" in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 6:
                    continue
                sym = parts[1].replace("sh","").replace("sz","").replace("bj","").zfill(6)
                try:
                    close = float(parts[4])
                except ValueError:
                    continue
                if sym not in result:
                    result[sym] = []
                result[sym].append(close)
        except FileNotFoundError:
            logger.error(f"npx 未找到于 {_NPX}，请确认 WorkBuddy Node.js 安装")
            break
        except Exception as e:
            logger.warning(f"westock kline 批次失败: {e}")
            continue

    logger.info(f"westock batch kline: {len(result)}/{len(codes)} 只有效")
    return result


def batch_change_pct(codes: list[str]) -> dict[str, float]:
    """批量获取涨跌幅。返回 {code: change_pct%}。"""
    prices = batch_kline(codes, limit=2)
    changes = {}
    for code, closes in prices.items():
        if len(closes) >= 2 and closes[1] > 0:
            changes[code] = round((closes[0] / closes[1] - 1) * 100, 2)
    return changes


def batch_close_prices(codes: list[str]) -> dict[str, float]:
    """批量获取当日收盘价。返回 {code: close_price}。"""
    prices = batch_kline(codes, limit=1)
    return {c: v[0] for c, v in prices.items() if v}
