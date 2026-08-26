"""
walk_forward_oos_m2.py — lvrev 策略 M2: 滚动窗口权重再估计 (消除权重选择偏差)

动机(来自 v4.26 OOS 报告残项):
  固定权重 W_DEFAULT = {vol:0.45, rev:0.35, q:0.12, value:0, g:0} 取自"全周期 IC 研究",
  本质上是从同一批数据里选出来的 -> 残留**选择偏差**。如果策略真实有效,
  那么用"决策时仅已知的历史"重估权重, OOS 表现应 ≈ 固定权重。否则说明 alpha
  依赖于事后调权(过拟合的另一种形式)。

方法(严格 point-in-time):
  - 复用 walk_forward_oos 的 load_data / run_strategy (同一组合模拟与闸门, 保证可比)
  - 每个 OOS 折页 y(2021~2026): 训练窗 = [y-3 年起, y-1 年末] 的全部交易日
      逐训练日截面计算 低波/反转/质量/成长 各自对未来20日收益的 Rank-IC -> 均值
      权重 = 对 {vol,rev,q} 做 max(0, meanIC) 归一化 (growth/value 保持策略既定 0)
  - 用该折重估权重交易 OOS 年, 与"同引擎固定权重 OOS"逐折配对对比
  - 另给出全周期(2020~2025)IC 加权权重, 与 W_DEFAULT 直接对照, 看固定权重是否本就
    近似 IC 最优 -> 若接近, 则重估不会系统性改变结果, 偏差本就小

判读:
  - 滚动权重 OOS 几何收益 ≈ 固定权重 OOS (+ 年均差异小) -> 权重选择偏差已被排除
  - 若滚动显著优于/劣于固定 -> 说明固定权重是"巧合/过拟合", 需改用滚动方案实盘

依赖: walk_forward_oos.py (同目录), src/lvrev_scorer (已支持 weights 参数)

用法:
  python walk_forward_oos_m2.py
"""
from __future__ import annotations
import sys, os, json, time, logging
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))  # 复用 walk_forward_oos 的 load_data/run_strategy
from lvrev_scorer import factor_scores  # noqa: E402
from walk_forward_oos import (  # noqa: E402
    load_data, run_strategy, oos_forward_ic, INITIAL_CAPITAL, DATA_END,
)
from lvrev_scorer import W_DEFAULT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("wfo_m2")

# OOS 折页年 + 训练窗(y-3 .. y-1)
FOLD_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
TRAIN_LOOKBACK_YEARS = 3
MIN_TRAIN_DAYS = 120  # 训练日不足则回退固定权重


def _add_fwd20(px: pd.DataFrame) -> pd.DataFrame:
    """前向20日收益(仅分析用, 不进决策); per-code 正确平移, 按索引对齐。"""
    fwd = (px.groupby("code")["close"].transform(lambda s: s.shift(-20))
           / px["close"] - 1.0)
    px = px.copy()
    px["fwd20"] = fwd
    return px


def estimate_weights(px: pd.DataFrame, day_frames: dict,
                     start: str, end: str) -> dict | None:
    """训练窗内逐日截面 Rank-IC -> 对 {vol,rev,q} 做 IC 加权归一化。

    返回 {weights, mean_ic, n_days} 或 None(训练日不足, 调用方回退固定权重)。
    """
    ic_sum = {"low_vol": 0.0, "reversal": 0.0, "quality": 0.0, "growth": 0.0}
    n = 0
    sd, ed = pd.Timestamp(start), pd.Timestamp(end)
    for d, day in day_frames.items():
        if not (sd <= d <= ed):
            continue
        if len(day) < 200:
            continue
        fwd = day.get("fwd20")
        if fwd is None or fwd.notna().sum() < 100:
            continue
        fs = factor_scores(day)
        fr = fwd.rank()
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in ic_sum:
                ic = fs[k].rank().corr(fr, method="pearson")
                if pd.notna(ic):
                    ic_sum[k] += ic
        n += 1
    if n < MIN_TRAIN_DAYS:
        return None
    mean_ic = {k: ic_sum[k] / n for k in ic_sum}
    raw = {kk: max(0.0, mean_ic[kk]) for kk in ("low_vol", "reversal", "quality")}
    tot = sum(raw.values())
    if tot <= 0:
        return None
    w = {"vol": raw["low_vol"] / tot, "rev": raw["reversal"] / tot,
         "q": raw["quality"] / tot, "value": 0.0, "g": 0.0}
    return {"weights": w, "mean_ic": mean_ic, "n_days": n}


def main():
    t_total = time.time()
    px = load_data()
    px = _add_fwd20(px)
    all_dates = pd.to_datetime(sorted(px["date"].unique()))
    day_frames = {d: g for d, g in px.groupby("date")}

    # 全历史市场均线(折页起点也有完整 MA60)
    mkt_avg = px.groupby("date")["close"].mean().reindex(all_dates)
    mkt_ma60 = mkt_avg.rolling(60, min_periods=60).mean()

    # --- 全周期 IC 加权权重(对照 W_DEFAULT 来源) ---
    full_ic = estimate_weights(px, day_frames, "2020-01-01", "2025-12-31")
    LOG.info("全周期(2020-2025) IC 加权权重: %s (n=%s)",
             {k: round(v, 3) for k, v in full_ic["weights"].items()},
             full_ic["n_days"])
    LOG.info("  W_DEFAULT(固定): %s", {k: round(v, 3) for k, v in W_DEFAULT.items()})

    # --- 逐折: 固定权重 OOS vs 滚动权重 OOS ---
    LOG.info("=" * 70)
    LOG.info("M2: 逐折 固定权重 OOS vs 滚动权重 OOS")
    LOG.info("=" * 70)

    paired = {}
    for y in FOLD_YEARS:
        fs = f"{y}-01-01"
        fe = f"{y}-12-31" if y < 2026 else DATA_END
        # 固定权重(同引擎)
        r_fixed = run_strategy(px, day_frames, mkt_avg, mkt_ma60, fs, fe,
                               weight_override=None)
        # 训练窗 + 重估权重
        ts = f"{y - TRAIN_LOOKBACK_YEARS}-01-01"
        te = f"{y - 1}-12-31"
        est = estimate_weights(px, day_frames, ts, te)
        if est is None:
            LOG.warning("  %s 训练日不足, 回退固定权重", y)
            w = None
            r_roll = r_fixed
            mean_ic = None
        else:
            w = est["weights"]
            r_roll = run_strategy(px, day_frames, mkt_avg, mkt_ma60, fs, fe,
                                  weight_override=w)
            mean_ic = est["mean_ic"]
        paired[str(y)] = {
            "fixed": r_fixed, "rolling": r_roll,
            "weights": w, "train_mean_ic": mean_ic,
            "train_window": f"{ts}~{te}",
        }
        fw = r_fixed.get("return_pct", float("nan"))
        rw = r_roll.get("return_pct", float("nan"))
        print(f"  OOS {y}: 固定 {fw:+.2f}% | 滚动 {rw:+.2f}% | "
              f"权重 vol/rev/q="
              f"{('-' if w is None else round(w['vol'],2))}/"
              f"{('-' if w is None else round(w['rev'],2))}/"
              f"{('-' if w is None else round(w['q'],2))} | "
              f"夏普 固定 {r_fixed.get('sharpe','-')} / 滚动 {r_roll.get('sharpe','-')}")

    # --- 几何聚合对比 ---
    def geo(fold_key):
        yrs = [y for y in FOLD_YEARS if paired[str(y)][fold_key]]
        rets = [1 + paired[str(y)][fold_key]["return_pct"] / 100 for y in yrs]
        avg = float(np.mean([paired[str(y)][fold_key]["return_pct"] for y in yrs]))
        return float(np.prod(rets) - 1) * 100, avg

    g_fix, a_fix = geo("fixed")
    g_roll, a_roll = geo("rolling")
    print(f"\n  >> OOS 几何聚合 固定: {g_fix:+.2f}% | 年均 {a_fix:+.2f}%")
    print(f"  >> OOS 几何聚合 滚动: {g_roll:+.2f}% | 年均 {a_roll:+.2f}%")
    print(f"  >> 差异(滚动-固定): 几何 {g_roll - g_fix:+.2f}pp | 年均 "
          f"{a_roll - a_fix:+.2f}pp")

    # --- OOS 前向 Rank-IC(与 M1 同口径, 供交叉验证) ---
    ic = oos_forward_ic(px)

    results = {
        "full_period_ic_weights": full_ic["weights"] if full_ic else None,
        "full_period_mean_ic": full_ic["mean_ic"] if full_ic else None,
        "w_default": W_DEFAULT,
        "paired": {y: {
            "fixed": paired[y]["fixed"], "rolling": paired[y]["rolling"],
            "weights": paired[y]["weights"],
            "train_mean_ic": paired[y]["train_mean_ic"],
            "train_window": paired[y]["train_window"],
        } for y in paired},
        "oos_geo_fixed_pct": round(g_fix, 2),
        "oos_avg_fixed_pct": round(a_fix, 2),
        "oos_geo_rolling_pct": round(g_roll, 2),
        "oos_avg_rolling_pct": round(a_roll, 2),
        "oos_ic": ic,
    }
    out = HERE / "walk_forward_oos_m2_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\n结果已落盘: {out}  ({time.time()-t_total:.1f}s)")
    print(f"结论: 滚动权重 OOS 与固定权重 OOS 差异 "
          f"{g_roll-g_fix:+.2f}pp(几何) / {a_roll-a_fix:+.2f}pp(年均) "
          f"-> {'偏差已排除' if abs(g_roll-g_fix) < 5 else '需关注'}")


if __name__ == "__main__":
    main()
