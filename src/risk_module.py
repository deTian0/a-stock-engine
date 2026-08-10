"""
risk_module.py — 交易风控辅助: 建议止损价、仓位调整、板块集中度、流动性检查

借鉴 vnpy risk management 模块思路:
  - ATR 止损: entry - 2×ATR
  - Kelly 仓位: base × edge/(edge+risk)
  - 板块集中度: 同板块不超过3只
  - 流动性: 日均成交额 > min_amount
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


def enrich_risk_metrics(l4_results: pd.DataFrame, regime_cap: float = 0.5,
                        total_capital: float = 100000) -> pd.DataFrame:
    """
    给 L4 结果加风控指标: 止损价、调整后仓位、流动性标签、板块警告。
    """
    df = l4_results.copy()
    if len(df) == 0:
        return df

    # ---- 1. ATR 止损价 ----
    try:
        from westock_helpers import batch_kline
        codes = df["code"].astype(str).str.zfill(6).unique().tolist()
        # 取21天K线算ATR
        kdata = batch_kline(codes, limit=21)
        
        stops = {}
        for code, closes in kdata.items():
            if len(closes) < 15:
                continue
            # ATR = max(high-low, high-prev_close, prev_close-low) simplified
            tr = [abs(closes[i] - closes[i+1]) for i in range(len(closes)-1) if closes[i+1] > 0]
            if tr:
                atr = sum(tr[-14:]) / min(14, len(tr))
                entry = closes[0]  # current close
                stops[code] = {
                    "atr": round(atr, 2),
                    "stop_loss": round(entry - 2 * atr, 2),
                    "stop_pct": round(-2 * atr / entry * 100, 1) if entry > 0 else 0,
                }

        df["atr14"] = df["code"].astype(str).str.zfill(6).map(lambda c: stops.get(c, {}).get("atr", 0))
        df["stop_loss"] = df["code"].astype(str).str.zfill(6).map(lambda c: stops.get(c, {}).get("stop_loss", 0))
        df["stop_pct"] = df["code"].astype(str).str.zfill(6).map(lambda c: stops.get(c, {}).get("stop_pct", 0))
        logger.info(f"ATR止损: {len(stops)} 只")
    except Exception as e:
        logger.warning(f"ATR止损计算异常: {e}")

    # ---- 2. Kelly 仓位调整 ----
    # 简化版: position = 基准 × (score/100) × (1/√ATR%)
    df["suggested_position"] = df.get("suggested_position", regime_cap * 100)
    if "composite_score" in df.columns and "atr14" in df.columns:
        base = regime_cap * 100  # 基准仓位%
        for i, row in df.iterrows():
            score = row.get("composite_score", 60)
            atr14 = row.get("atr14", 0)
            close = row.get("close", 0)
            if close > 0 and atr14 > 0:
                atr_pct = atr14 / close  # ATR相对价格
                # 高波动 → 小仓位, 高评分 → 大仓位
                vol_factor = min(1.0, 0.02 / max(atr_pct, 0.005))  # 2%波动率基准
                score_factor = score / 60  # 60分基准
                kelly_adj = base * score_factor * vol_factor * 0.5  # half-Kelly
                df.at[i, "suggested_position"] = round(min(kelly_adj, regime_cap * 100 * 0.5), 1)

    # ---- 3. 板块集中度 ----
    if "sector" in df.columns:
        sector_counts = Counter(df["sector"].dropna())
        df["sector_warning"] = df["sector"].map(
            lambda s: f"⚠️同板块{count}只" if (count := sector_counts.get(s, 0)) >= 3 else ""
        )

    # ---- 4. 流动性标签 ----
    if "amount" in df.columns:
        df["liquidity_tag"] = df["amount"].apply(
            lambda a: "🟢高" if a > 5e8 else ("🟡中" if a > 2e8 else ("🔴低" if a > 5e7 else "⚠️极低"))
        )

    return df


def allocate_basket(scores: list, sleeve_budget: float,
                    method: str = "score_weighted", max_single: float = None) -> list:
    """
    篮子分配: 把 sleeve_budget (占总资金的比例, 0~1) 分配到 N 只标的。

    返回每只标的「占总资金的比例」列表, 求和严格等于 sleeve_budget。
    这是修复「单只票按 position_cap 满算 → 多只累加远超上限」的核心函数。

    参数:
      scores       : 各标的评分列表 (越高权重越大)
      sleeve_budget: 该篮子预算, 占总资金比例 (如 0.15 = 总资金 15%)
      method       : "score_weighted" 按评分权重 | "equal" 等权
      max_single   : 单只占总资金硬上限 (如 0.08); 超限截断后余量再分给其它票

    边界:
      scores 为空 -> []
      sleeve_budget <= 0 -> 全 0
      method 非 score_weighted 或评分全 <=0 -> 退化为等权
    """
    n = len(scores)
    if n == 0:
        return []
    if sleeve_budget <= 0:
        return [0.0] * n

    if method != "score_weighted":
        return [sleeve_budget / n] * n

    s = [max(float(x), 0.0) for x in scores]
    total = sum(s)
    if total <= 0:
        return [sleeve_budget / n] * n

    alloc = [x / total * sleeve_budget for x in s]

    # 单只硬上限: 截断超限部分, 余量按比例补给未超限的票
    if max_single is not None and max_single > 0:
        capped = [a > max_single for a in alloc]
        if any(capped):
            excess = sum(a - max_single for a, c in zip(alloc, capped) if c)
            for i in range(n):
                if capped[i]:
                    alloc[i] = max_single
            free_idx = [i for i in range(n) if not capped[i]]
            free_sum = sum(alloc[i] for i in free_idx)
            if free_idx:
                if free_sum > 0:
                    for i in free_idx:
                        alloc[i] += excess * (alloc[i] / free_sum)
                else:
                    add = excess / len(free_idx)
                    for i in free_idx:
                        alloc[i] += add

    # 浮点修正: 归一化确保精确求和 = sleeve_budget
    s_sum = sum(alloc)
    if s_sum > 0:
        alloc = [a * sleeve_budget / s_sum for a in alloc]
    return alloc
