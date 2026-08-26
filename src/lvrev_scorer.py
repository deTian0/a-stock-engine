"""lvrev_scorer — a-stock-engine 的唯一选股 alpha 内核(低波 + 反转 + 质量 + 成长 + 价值)。

这是回测(local_backtest._score_lowvol_rev) 与 盘前实盘(multifactor) 共用的
**单一事实来源**: 两者都调用本模块, 保证"研究成果"与"实盘选股"跑的是同一套数学,
不再出现"回测证明买弱/低波胜、实盘却仍在跑被证伪的反 alpha 动量内核"的错位。

内核设计(依据 alpha_research.py 诊断, 详见 STRATEGY.md §7):
  - 低波动 = 唯一稳健正 alpha(IC+0.11, 多空+23%/yr)
  - 反转(近N日超跌) = 把"反 alpha"的动量因子反向使用(买弱不买强)
  - 质量(roe/负债率) / 成长(营收增速) ≈ 零 alpha, 仅作稳定器
  - 价值(bp=1/pb + sp=1/ps_ttm) 为第二正 alpha, 但 long-only 集中持仓下净拖累,
    故默认关(VALUE_FACTOR=False); ey(1/pe_ttm) 为更弱的盈利维度, 加性可选。

对外接口:
  - score_lvrev(df, value_factor, ey_weight) -> df(新增 composite_score)
      评分只消费列: vol20, rev_chg(或 chg_20d/chg_10d 兜底), roe, debt_ratio,
      revenue_growth, pb, ps_ttm, pe(仅 ey 用)。
  - apply_entry_gates(df, reversal_q) -> pd.Series(bool)
      入场风控闸门(镜像 local_backtest.run_portfolio 的 lvrev 分支):
      f_trend 硬闸门(MA20>MA60) + f_rs 软闸门(不接飞刀) + 近N日超跌(底部 reversal_q 分位)
      + 长线支撑(MA60) + 低波门槛(≤当日波动中位数)。
      需列: rev_chg, vol20, close, ma20, ma60(或 trend_up), rs20。
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# 默认评分权重(v4.26: 成长 g 清零 — 营收增速 Rank-IC=-0.030/t=-1.62 弱负且不显著,
#   属纯噪声/负贡献; 隔离回测 v4.25(+10.14%/夏普0.19) -> v4.26(+28.59%/夏普0.40),
#   全部指标同向改善。任何权重倾斜(vol/rev 增减)均不如"仅清零成长"的极简改动)
W_DEFAULT = dict(vol=0.45, rev=0.35, value=0.0, q=0.12, g=0.0)
W_VALUE = dict(vol=0.38, rev=0.27, value=0.18, q=0.10, g=0.0)


def factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """各因子的 point-in-time 归一化得分(均映射到 [0,1], 且方向已对齐预期收益)。

    返回列: low_vol(波动越低分越高) / reversal(越超跌分越高) / quality(杠杆越低分越高)
            / growth(营收增速越高分越高) / value(越便宜分越高)。
    这是权重重估(M2)与组合评分共用的事实来源: 调整权重只需缩放这些得分,
    不改变因子方向语义, 杜绝"重写评分逻辑"导致的漂移。
    """
    d = df
    n = len(d)

    # 1) 低波动: 波动率越低越好 -> (1 - 排名)
    if "vol20" in d.columns:
        s_vol = d["vol20"].rank(pct=True, na_option="keep")
    else:
        s_vol = pd.Series(0.5, index=d.index)
    low_vol = (1 - s_vol.fillna(0.5))

    # 2) 反转: 近N日越超跌越买(均值回归) — rev_chg 已由调用方按窗口算好
    if "rev_chg" in d.columns:
        s_rev = (-d["rev_chg"]).rank(pct=True, na_option="keep")
    elif "chg_20d" in d.columns:
        s_rev = (-d["chg_20d"]).rank(pct=True, na_option="keep")
    elif "chg_10d" in d.columns:
        s_rev = (-d["chg_10d"]).rank(pct=True, na_option="keep")
    else:
        s_rev = pd.Series(0.5, index=d.index)
    reversal = s_rev.fillna(0.5)

    # 3) 质量稳定器: 仅保留低杠杆(debt_ratio), 不再用 ROE 正向排序
    #    (ROE Rank-IC=-0.078/t=-2.94 显著负, 见#23实证; 高ROE是负alpha,
    #     用作正向排序偏置会系统性选中跑输股。v4.25: 移除 ROE 排序项, B 项隔离回测
    #     +0.59%->+10.14%; 注意 W_DEFAULT.q 不可同时下调, 否则与去ROE交互崩至-5.51%)
    s_q = pd.Series(0.5, index=d.index)
    if "debt_ratio" in d.columns:
        s_q = s_q - (d["debt_ratio"].clip(0, 100).fillna(50) / 100.0)
    s_q = s_q.rank(pct=True, na_option="keep").fillna(0.5)
    quality = s_q

    # 4) 成长: 营收增速越高越好(零 alpha, 稳定器)
    if "revenue_growth" in d.columns:
        s_g = d["revenue_growth"].clip(-50, 100).fillna(0).rank(pct=True, na_option="keep")
    else:
        s_g = pd.Series(0.5, index=d.index)
    growth = s_g.fillna(0.5)

    # 5) 价值(便宜): bp=1/pb(book yield) + sp=1/ps_ttm(sales yield)
    #    bp 权重高于 sp(0.6/0.4), 与诊断强度一致。VALUE_FACTOR=False 时退化为 0。
    if "pb" in d.columns and "ps_ttm" in d.columns:
        bp_clean = d["pb"].where(d["pb"] > 0)
        sp_clean = d["ps_ttm"].where(d["ps_ttm"] > 0)
        bp_inv = 1.0 / bp_clean
        sp_inv = 1.0 / sp_clean
        s_val_bp = bp_inv.rank(pct=True, na_option="keep").fillna(0.5)
        s_val_sp = sp_inv.rank(pct=True, na_option="keep").fillna(0.5)
        value = (0.6 * s_val_bp + 0.4 * s_val_sp).fillna(0.5)
    else:
        value = pd.Series(0.5, index=d.index)

    return pd.DataFrame({
        "low_vol": low_vol, "reversal": reversal, "quality": quality,
        "growth": growth, "value": value,
    }, index=d.index)


def score_lvrev(df: pd.DataFrame, value_factor: bool = False,
                ey_weight: float = 0.0, weights: dict | None = None
                ) -> pd.DataFrame:
    """低波+反转+质量+成长(+可选价值/ey) 评分内核。返回带 composite_score 的 df。

    与 local_backtest._score_lowvol_rev 数学等价; 行为由调用方传入的
    value_factor / ey_weight / weights 控制。
      - 默认 weights=None -> 用 W_VALUE/W_DEFAULT(还原 +26.1% 基线权重)。
      - 传入 weights(dict) -> 按自定义系数缩放 factor_scores(供 M2 滚动权重重估)。
    权重键: vol / rev / value / q / g(缺失键按 0 计)。
    """
    if len(df) == 0:
        df = df.copy()
        df["composite_score"] = 0.0
        return df
    d = df.copy()

    fs = factor_scores(d)

    # 6) ey(盈利收益率 1/pe_ttm) 加性价值维度(可选): 不稀释低波/反转,
    #    仅当 ey_weight>0 时作为额外加分项(1/pe 越高=越便宜)。
    if ey_weight > 0 and "pe" in d.columns:
        pe_clean = d["pe"].where((d["pe"] > 0) & (d["pe"] < 300))
        ey_rank = (1.0 / pe_clean).rank(pct=True, na_option="keep").fillna(0.5)
    else:
        ey_rank = 0.0

    if weights is None:
        w = dict(W_VALUE) if value_factor else dict(W_DEFAULT)
    else:
        w = {k: float(v) for k, v in weights.items()}

    d["composite_score"] = (
        w.get("vol", 0.0) * fs["low_vol"]
        + w.get("rev", 0.0) * fs["reversal"]
        + w.get("value", 0.0) * fs["value"]
        + w.get("q", 0.0) * fs["quality"]
        + w.get("g", 0.0) * fs["growth"]
        + ey_weight * ey_rank
    )
    return d.sort_values("composite_score", ascending=False)


def apply_entry_gates(df: pd.DataFrame, reversal_q: float = 0.30) -> pd.Series:
    """入场风控闸门(镜像 local_backtest.run_portfolio 的 lvrev 分支)。

    返回 bool 掩码: True=通过(可买), False=被拒。
    需列: rev_chg, vol20, close, ma20, ma60(或 trend_up), rs20(可选)。
    逻辑:
      - f_trend 硬闸门: MA20 > MA60(长趋势向上)
      - f_rs 软闸门: 价格 >= MA20*0.93(不接自由落体飞刀)
      - 近N日超跌: rev_chg <= 当日全市场底部 reversal_q 分位(反转核心判据)
      - 长线支撑: 价格 >= MA60*0.93
      - 低波门槛: vol20 <= 当日全市场波动中位数(只买"安静票")
    """
    if len(df) == 0:
        return pd.Series([], dtype=bool)

    keep = pd.Series(True, index=df.index)

    # 低波内核门槛: 当日全市场波动率中位数
    vol_med = (float(df["vol20"].median())
               if "vol20" in df.columns and df["vol20"].notna().any() else float("inf"))
    # 近N日超跌门槛: 全市场 rev_chg 底部 reversal_q 分位
    chg_q = (float(df["rev_chg"].quantile(reversal_q))
             if "rev_chg" in df.columns and df["rev_chg"].notna().any() else -0.05)

    for idx, row in df.iterrows():
        close = row.get("close")
        ma20 = row.get("ma20")
        ma60 = row.get("ma60")
        vol = row.get("vol20")
        rev = row.get("rev_chg")

        # f_trend 硬闸门
        if ma20 is not None and ma60 is not None:
            if ma20 <= ma60:
                keep[idx] = False
                continue
        # f_rs 软闸门(不接飞刀)
        if ma20 is not None and close is not None and ma20 > 0:
            if close < ma20 * 0.93:
                keep[idx] = False
                continue
        # 近N日超跌(反转核心)
        if rev is not None and not pd.isna(rev) and rev > chg_q:
            keep[idx] = False
            continue
        # 长线支撑
        if ma60 is not None and close is not None and ma60 > 0:
            if close < ma60 * 0.93:
                keep[idx] = False
                continue
        # 低波门槛
        if vol is not None and not pd.isna(vol) and vol_med != float("inf"):
            if vol > vol_med:
                keep[idx] = False
                continue

    return keep
