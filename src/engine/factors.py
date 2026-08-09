"""
factor_engine.py — 统一因子引擎（v2.1）

所有模块共享的因子计算逻辑。单一入口，消除代码重复。

用法:
    from engine.factors import score_stocks

    scored = score_stocks(df, weights=None)
    picks = pick_top_by_sector(scored, max_per_sector=5)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认权重配置（可被 config.yaml 覆盖）
DEFAULT_WEIGHTS = {
    "momentum": 0.15,
    "momentum_25d": 0.10,
    "momentum_multi": 0.10,
    "pe_value": 0.15,
    "pb_value": 0.10,
    "small_cap": 0.20,
    "low_vol": 0.10,
    "quality": 0.10,
}


def score_stocks(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    多因子评分 v2 — Z-Score 行业中性化。

    Parameters:
        df: DataFrame, 至少包含 sector 列 + 因子列
        weights: 因子权重，默认使用 DEFAULT_WEIGHTS

    Returns:
        DataFrame (排序后，新增 composite_score 列)
    """
    if len(df) == 0:
        return df
    if weights is None:
        weights = DEFAULT_WEIGHTS

    s = df.copy()
    s["sector"] = s["sector"].fillna("其他")

    # === Factor 1: 动量 (当日涨幅) ===
    s["f_momentum"] = _zscore(s, "change_pct", direction=1, default_col="chg_3d")

    # === Factor 2: 中期动量 (25日涨幅) ===
    s["f_momentum_25d"] = _zscore(s, "chg_25d", direction=1, default_col="chg_10d")

    # === Factor 3: 多周期动量 (3/6/10日均值) ===
    mom_cols = [c for c in ["chg_3d", "chg_6d", "chg_10d"] if c in s.columns]
    if mom_cols:
        s["_mom_avg"] = s[mom_cols].mean(axis=1, skipna=True)
        s["f_momentum_multi"] = _zscore(s, "_mom_avg", direction=1)

    # === Factor 4: PE 价值 (越低越好) ===
    if "pe" in s.columns:
        pe_clean = s["pe"].clip(0, 300).where(s["pe"] > 0, np.nan)
        s["f_pe_value"] = _zscore(s.assign(_pe=pe_clean), "_pe", direction=-1)

    # === Factor 5: PB 价值 ===
    if "pb" in s.columns:
        pb_clean = s["pb"].clip(0, 50).where(s["pb"] > 0, np.nan)
        s["f_pb_value"] = _zscore(s.assign(_pb=pb_clean), "_pb", direction=-1)

    # === Factor 6: 小市值 ===
    s["f_small_cap"] = _zscore(s, "market_cap", direction=-1, default_col="float_cap")

    # === Factor 7: 低波动 ===
    vol_cols = [c for c in ["amplitude", "turnover"] if c in s.columns]
    if vol_cols:
        s["_vol_proxy"] = s[vol_cols].mean(axis=1, skipna=True)
        s["f_low_vol"] = _zscore(s, "_vol_proxy", direction=-1)

    # === 综合评分 ===
    factor_cols = [c for c in s.columns if c.startswith("f_")]
    if not factor_cols:
        s["composite_score"] = 0.0
        return s

    s["raw_score"] = 0.0
    for col in factor_cols:
        w = weights.get(col, 1.0 / len(factor_cols))
        if col in s.columns:
            s["raw_score"] += s[col].fillna(0) * w

    # 行业中性化
    s["composite_score"] = s["raw_score"] - s.groupby("sector")["raw_score"].transform("mean")

    return s.sort_values("composite_score", ascending=False)


def pick_top_by_sector(df: pd.DataFrame, max_per_sector: int = 5,
                       min_score: float = None) -> list[dict]:
    """
    按板块分组选 Top N。

    Returns:
        [{sector, code, name, score, close, pe, market_cap, rank}, ...]
    """
    if len(df) == 0:
        return []
    if "composite_score" not in df.columns:
        df = score_stocks(df)

    if "sector" not in df.columns:
        # 无板块信息，直接取 Top
        picks = []
        for rank, (_, row) in enumerate(df.head(max_per_sector * 20).iterrows(), 1):
            picks.append(_row_to_pick(row, rank))
        return picks

    picks = []
    for sector, group in df.groupby("sector"):
        top = group.head(max_per_sector)
        if min_score is not None:
            top = top[top["composite_score"] >= min_score]
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            picks.append(_row_to_pick(row, rank, sector))

    return picks


def filter_candidates(df: pd.DataFrame,
                      min_market_cap: float = 20e8,
                      max_pe: float = 300,
                      exclude_st: bool = True) -> pd.DataFrame:
    """L2 过滤（所有模块共享）。"""
    c = df.copy()
    if exclude_st and "name" in c.columns:
        c = c[~c["name"].str.contains(r"ST|\*ST|退", na=False, regex=True)]
    if "market_cap" in c.columns:
        c = c[c["market_cap"] >= min_market_cap]
    if "pe" in c.columns:
        c = c[(c["pe"] > 0) & (c["pe"] <= max_pe)]
    if "close" in c.columns:
        c = c[c["close"] > 0]
    return c


# ===== 内部辅助 =====

def _zscore(df: pd.DataFrame, col: str, direction: int = 1,
            default_col: str = None) -> pd.Series:
    """行业分组 Z-Score 标准化。col 不存在时尝试 default_col。"""
    if col not in df.columns:
        if default_col and default_col in df.columns:
            col = default_col
        else:
            return pd.Series(0.0, index=df.index)

    group_means = df.groupby("sector")[col].transform("mean")
    group_stds = df.groupby("sector")[col].transform("std").clip(lower=0.001)
    z = ((df[col].fillna(group_means) - group_means) / group_stds).fillna(0)
    return z.clip(-3, 3) * direction


def _row_to_pick(row, rank: int, sector: str = None) -> dict:
    return {
        "sector": sector or str(row.get("sector", "未知")),
        "code": str(row.get("code", "")),
        "name": str(row.get("name", "")),
        "score": round(float(row.get("composite_score", 0)), 2),
        "close": float(row.get("close", 0)) if pd.notna(row.get("close")) else 0,
        "pe": float(row.get("pe", 0)) if pd.notna(row.get("pe")) else 0,
        "market_cap": float(row.get("market_cap", 0)) if pd.notna(row.get("market_cap")) else 0,
        "rank": rank,
    }
