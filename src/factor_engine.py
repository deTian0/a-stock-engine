"""
factor_engine.py — 统一因子引擎（v2.1）

所有模块共享的因子计算逻辑。单一入口，消除代码重复。

用法:
    from factor_engine import score_stocks

    scored = score_stocks(df, weights=None)
    picks = pick_top_by_sector(scored, max_per_sector=5)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认权重配置（可被 config.yaml 覆盖）
# 注意: key 必须与下方 f_ 因子列名一致! 旧版 config 用 momentum/pe_value 等无前缀名,
#       导致 score_stocks 中 weights.get(col) 永远命中不到 → 全部回退 1/len 均权,
#       调参彻底失效 (已修正 config 同步为 f_ 前缀)。
# 设计: 相对强度(f_rs 0.18) + 趋势(f_trend 0.10) 主导, 估值(pe+pb 0.17)显著下调,
#       质量降至0.05(规避 static 前视接盘), 见 B1/M6。
DEFAULT_WEIGHTS = {
    "f_momentum": 0.06,
    "f_momentum_25d": 0.05,
    "f_momentum_multi": 0.03,
    "f_rs": 0.18,           # 相对强度(价格站上MA20) — 新增, 治 4/6 月反向
    "f_trend": 0.10,        # 趋势(MA20>MA60) — 新增
    "f_pe_value": 0.10,     # 估值下调(原0.12)
    "f_pb_value": 0.07,     # 估值下调(原0.08)
    "f_small_cap": 0.06,
    "f_low_vol": 0.10,
    "f_quality": 0.05,
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

    # === Factor 1: 动量 (10日涨幅, 替代原当日涨幅) ===
    # 根因: 原用 change_pct(当日涨幅) 方差极大, 其 z-score 淹没有估值/质量因子,
    # 导致无论怎么调权重选股都退化成"买昨日涨幅王"(典型追涨杀跌, 胜率差)。
    # 改用 chg_10d(10日) 降低1日噪声主导, 让权重配置真正生效。
    s["f_momentum"] = _zscore(s, "chg_10d", direction=1, default_col="chg_25d")

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

    # === Factor 8: 质量 (roe/毛利率/负债率/营收增速/净利增速) ===
    # M3: 让 DEFAULT_WEIGHTS 的 quality(0.22) 真正生效; 列缺失时自动跳过(权重不计入)
    qual_cols = [c for c in ["roe", "gross_margin", "debt_ratio",
                             "revenue_growth", "profit_growth"] if c in s.columns]
    if qual_cols:
        q_w = {"roe": 0.30, "gross_margin": 0.20, "debt_ratio": 0.20,
               "revenue_growth": 0.15, "profit_growth": 0.15}
        q_total = sum(q_w.get(c, 0) for c in qual_cols)
        q_parts = []
        for c in qual_cols:
            # 负债率越低越好(反向); 其余越高越好
            direction = -1 if c == "debt_ratio" else 1
            q_parts.append(_zscore(s, c, direction=direction) * q_w[c])
        s["f_quality"] = sum(q_parts) / q_total

    # === Factor 9: 相对强度 (价格站上 MA20) — B1 新增, 治 4/6 月反向 ===
    # rs20 = (close/MA20 - 1)*100, 由 load_day_data 注入; 站上均线=强势, 越高越好。
    # 与纯动量(chg_10d)区别: RS 是"相对自身均线位置", 过滤掉已见顶回落的票,
    # 实现"只买站上均线的强势股"的择时意图, 而非追当日涨幅王。
    if "rs20" in s.columns:
        s["f_rs"] = _zscore(s, "rs20", direction=1, default_col="rs60")

    # === Factor 10: 趋势 (MA20 > MA60) — B1 新增 ===
    # 多头排列(中短期均线上方)给 +1, 否则 -1。直接映射, 不依赖 z-score 分布。
    if "trend_up" in s.columns:
        s["f_trend"] = s["trend_up"].map({True: 1.0, False: -1.0}).fillna(0.0)

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
