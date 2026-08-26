"""
westock_helpers.py — westock-data CLI 辅助函数（共享模块）

使用绝对路径 npx，避免 PATH 缺失问题。
"""

import subprocess
import logging
from pathlib import Path
from collections import defaultdict

from sys_config import get_encoding, get_npx_path

logger = logging.getLogger(__name__)
_NPX = get_npx_path()
_ENCODING = get_encoding()



def _to_ws(code: str) -> str:
    """6位码 → sh/sz/bj 前缀（正确处理 ETF/基金代码）。

    沪市: 6xxxxx 股票(60/68/90)、5xxxxx 与 11xxxxx ETF/基金/债券 -> sh
    深市: 0xxxxx/3xxxxx 股票、1xxxxx ETF/基金(12/13/15/16/18) -> sz
    北交所: 8xxxxx/4xxxxx -> bj

    修复: 旧实现把 15xxxx/51xxxx 等 ETF 代码错判为 bj(北交所),
    导致 batch_quotes/batch_kline 对持仓 ETF 取不到实时价(查 bj 前缀无数据),
    持仓追踪 live 价修复在生产环境静默失效。
    """
    code = str(code).zfill(6)
    if code.startswith(("6", "9", "5", "11")):
        return f"sh{code}"
    if code.startswith(("0", "3", "2", "1")):
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
                capture_output=True, text=True, timeout=60,
                encoding=_ENCODING, errors="replace"
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


def batch_quotes(codes: list[str]) -> dict[str, dict]:
    """
    批量获取当日实时行情。返回 {code: {close, change_pct, amount, volume, volume_ratio, amplitude}}。
    通过 kline limit=2 拿今日+昨日，推导涨跌幅/量比/振幅（westock 的 exchange 列是换手率，非涨跌幅）。
    用于新浪/回退股票列表缺失实时字段时的全市场补全。
    """
    from collections import defaultdict

    result = {}
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        ws_codes = ",".join(_to_ws(c) for c in batch)
        try:
            r = subprocess.run(
                [_NPX, "-y", "westock-data-skillhub@1.0.5",
                 "kline", ws_codes, "--period", "day", "--limit", "2"],
                capture_output=True, text=True, timeout=90,
                encoding=_ENCODING, errors="replace"
            )
            if r.returncode != 0 or not r.stdout:
                continue
            # 列: | symbol | date | open | last | high | low | volume | amount | exchange |
            tmp = defaultdict(list)
            for line in r.stdout.strip().split("\n"):
                if "|" not in line or "symbol" in line or "Batch" in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 9:
                    continue
                sym = parts[1].replace("sh", "").replace("sz", "").replace("bj", "").zfill(6)
                try:
                    close = float(parts[4])   # last
                    high = float(parts[5])
                    low = float(parts[6])
                    volume = float(parts[7])
                    amount = float(parts[8])
                except (ValueError, IndexError):
                    continue
                tmp[sym].append((close, high, low, volume, amount))
            for sym, rows in tmp.items():
                if not rows:
                    continue
                close, high, low, volume, amount = rows[0]
                if len(rows) >= 2:
                    prev_close = rows[1][0]
                    prev_vol = rows[1][3]
                else:
                    prev_close = 0.0
                    prev_vol = 0.0
                chg = round((close / prev_close - 1) * 100, 2) if prev_close > 0 else 0.0
                vol_ratio = round(volume / prev_vol, 2) if prev_vol > 0 else 1.0
                amp = round((high - low) / prev_close * 100, 2) if prev_close > 0 else 0.0
                result[sym] = {
                    "close": close,
                    "change_pct": chg,
                    "amount": amount,
                    "volume": volume,
                    "volume_ratio": vol_ratio,
                    "amplitude": amp,
                }
        except FileNotFoundError:
            logger.error(f"npx 未找到于 {_NPX}，请确认 WorkBuddy Node.js 安装")
            break
        except Exception as e:
            logger.warning(f"westock quotes 批次失败: {e}")
            continue

    logger.info(f"westock batch quotes: {len(result)}/{len(codes)} 只")
    return result


def batch_tech_indicators(codes: list[str]) -> dict[str, dict]:
    """
    批量获取技术指标: MA均线 + MACD + RSI。
    返回 {code: {signal, strength, ma_signal, macd_signal, rsi_value, ...}}
    """
    result = {}
    batch_size = 50  # technical 输出字段多，批次小一点
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        ws_codes = ",".join(_to_ws(c) for c in batch)
        try:
            r = subprocess.run(
                [_NPX, "-y", "westock-data-skillhub@1.0.5",
                 "technical", ws_codes, "--indicator", "ma,macd,rsi"],
                capture_output=True, text=True, timeout=60,
                encoding=_ENCODING, errors="replace"
            )
            if r.returncode != 0 or not r.stdout:
                continue
            for line in r.stdout.strip().split("\n"):
                if "|" not in line or "code" in line or "Batch" in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 15:
                    continue
                sym = parts[1].replace("sh","").replace("sz","").replace("bj","").zfill(6)
                try:
                    close = float(parts[4])
                    ma20 = float(parts[7]) if len(parts) > 7 and parts[7] else 0
                    ma60 = float(parts[9]) if len(parts) > 9 and parts[9] else 0
                    dif = float(parts[16]) if len(parts) > 16 and parts[16] else 0
                    dea = float(parts[17]) if len(parts) > 17 and parts[17] else 0
                    macd_bar = float(parts[18]) if len(parts) > 18 and parts[18] else 0
                    rsi6 = float(parts[20]) if len(parts) > 20 and parts[20] else 50
                except (ValueError, IndexError):
                    continue

                # 均线信号
                ma_signals = []
                if ma20 > 0:
                    if close > ma20:
                        ma_signals.append("站上MA20")
                    else:
                        ma_signals.append("跌破MA20")
                if ma60 > 0:
                    if close > ma60:
                        ma_signals.append("多头")
                    else:
                        ma_signals.append("空头")

                # MACD信号
                if dif > dea:
                    macd_sig = "金叉" if macd_bar > 0 else "收敛"
                else:
                    macd_sig = "死叉" if macd_bar < 0 else "发散"

                # RSI信号
                if rsi6 > 75:
                    rsi_sig = "超买⚠️"
                elif rsi6 < 25:
                    rsi_sig = "超卖💡"
                else:
                    rsi_sig = "中性"

                # 综合信号词
                bullish = sum([1 for s in ma_signals if "站上" in s or "多头" in s]) + (1 if dif > dea else 0)
                if bullish >= 2:
                    signal = "🟢偏多"
                elif bullish == 0:
                    signal = "🔴偏空"
                else:
                    signal = "🟡震荡"

                result[sym] = {
                    "signal": signal,
                    "ma": ",".join(ma_signals) if ma_signals else "-",
                    "macd": macd_sig,
                    "rsi": f"{rsi6:.0f}({rsi_sig})",
                    "ma20": round(ma20, 2),
                    "ma60": round(ma60, 2),
                    "dif": round(dif, 2),
                    "dea": round(dea, 2),
                }
        except Exception as e:
            logger.warning(f"technical 批次失败: {e}")
            continue

    logger.info(f"技术指标: {len(result)}/{len(codes)} 只")
    return result


def batch_close_prices(codes: list[str]) -> dict[str, float]:
    """批量获取当日收盘价。返回 {code: close_price}。"""
    prices = batch_kline(codes, limit=1)
    return {c: v[0] for c, v in prices.items() if v}
