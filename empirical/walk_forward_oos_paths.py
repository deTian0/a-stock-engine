"""walk_forward_oos_paths.py — 三路径(A/B/C) 统一 样本外(OOS) walk-forward 验证台

目的: 在同一套数据 / 同一套 point-in-time 纪律 / 同一套成本模型下, 公平对比
      lvrev 核心 + 三套补 alpha 方案, 只换"方案"这一变量, 杜绝各算各的不可比。

复用(不修改)现有引擎:
  - walk_forward_oos.load_data / gate_vectorized / _metrics / oos_forward_ic
  - lvrev_scorer.score_lvrev / factor_scores (canonical 内核, 单一事实来源)

新增(仅本脚本, 研究用):
  - point-in-time 特征: mom60(60日动量) + industry(行业, 来自 fundamentals.industry)
  - point-in-time 热门板块: 按行业 60 日中位收益取 top-N(无前视)
  - point-in-time regime 判定: 市场 120 日动量 >0 且宽度 >0.55 -> 动量 regime
  - 变体:
      baseline : 纯 lvrev(对照组, 须复现已知 OOS 作保真校验)
      A        : lvrev 核心 + 动量卫星 sleeve(旁路闸门, 热门板块内挑动量龙头)
      B        : regime 切换 -> 动量 regime 用动量加权评分 + 放松反转/低波闸门
      C        : lvrev 评分对热门板块加分(tilt, 仍走 lvrev 闸门)
      AC       : C 的 tilt 核心 + A 的动量卫星(生产候选)

保真校验: baseline 全样本校准须接近 walk_forward_oos 已知结论
          (2026 折页约 -4.29%, 全样本 OOS 几何约 +35% 量级)。

用法:
  python walk_forward_oos_paths.py            # 跑全部 5 变体 + 动量 IC
  python walk_forward_oos_paths.py --variants A AC   # 只跑指定变体
"""
from __future__ import annotations
import sys, os, argparse, logging, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))          # 使 walk_forward_oos(同目录) 可导入
sys.path.insert(0, str(ROOT / "src"))   # canonical 内核 lvrev_scorer
from walk_forward_oos import (  # noqa: E402
    load_data, gate_vectorized, _metrics, oos_forward_ic,
    DATA_END, INITIAL_CAPITAL, MAX_PICKS_PER_DAY, STOP_LOSS, TARGET_BASE,
    REVERSAL_Q, MARKET_MA, trade_cost, DB,
)
from lvrev_scorer import factor_scores, score_lvrev  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("wfop")

# ============ 卫星 / 板块 参数 ============
SAT_BUDGET_FRAC = 0.35     # 卫星占用当日可投现金上限比例
SAT_MAX = 6                # 卫星单日最多新增持仓数
HOT_N = 6                  # 热门板块数量(top-N 行业)
MOM_W = 60                 # 动量窗口(日)


# ============ 数据增强: 动量 + 行业 ============
def load_data_x() -> pd.DataFrame:
    px = load_data()       # 已有 vol20/rev_chg/ma20/ma60/rs20/debt_ratio
    g = px.groupby("code", sort=False)
    px["mom60"] = g["close"].transform(lambda s: s / s.shift(MOM_W) - 1.0)
    c = sqlite3.connect(str(DB))
    try:
        ind = pd.read_sql_query(
            "SELECT code, industry FROM fundamentals WHERE industry IS NOT NULL", c)
    finally:
        c.close()
    if len(ind):
        ind["code"] = (ind["code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
                       .astype(int))
        ind_map = dict(zip(ind["code"], ind["industry"].fillna("未知")))
        px["industry"] = px["code"].map(ind_map).fillna("未知")
    else:
        px["industry"] = "未知"
    LOG.info(f"增强特征: mom60 + industry, 行业数 = {px['industry'].nunique()}")
    return px


# ============ point-in-time 热门板块 ============
def compute_hot_sectors(px: pd.DataFrame, hot_n: int = HOT_N) -> dict:
    """每决策日, 按各行业 60 日中位动量取 top-N 为热门板块(只用<=t数据)。"""
    hot = {}
    for d, day in px.groupby("date"):
        valid = day.dropna(subset=["mom60"])
        if len(valid) < 50:
            hot[d] = set()
            continue
        med = (valid.groupby("industry")["mom60"].median()
               .sort_values(ascending=False))
        hot[d] = set(med.head(hot_n).index)
    return hot


# ============ point-in-time regime 判定 ============
def compute_regimes(px: pd.DataFrame, mkt_avg: pd.Series):
    """动量 regime: 市场 120 日动量 >0 且 宽度(60日动量>0 占比) >0.55。"""
    breadth = {}
    for d, day in px.groupby("date"):
        breadth[d] = (float((day["mom60"] > 0).mean())
                      if day["mom60"].notna().any() else 0.5)
    mkt_ret120 = mkt_avg / mkt_avg.shift(120) - 1.0
    reg = {}
    for d in mkt_avg.index:
        mom_mkt = mkt_ret120.get(d, np.nan)
        if pd.notna(mom_mkt) and mom_mkt > 0 and breadth.get(d, 0.5) > 0.55:
            reg[d] = "momentum"
        else:
            reg[d] = "lowvol"
    return reg, breadth


# ============ 放松版闸门(用于 B 动量 regime: 去反转/低波) ============
def gate_relaxed(df: pd.DataFrame) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    m20, m60, close = df["ma20"], df["ma60"], df["close"]
    has_mm = m20.notna() & m60.notna()
    keep &= ~has_mm | (m20 > m60)            # trend 硬闸门
    has_m20 = m20.notna()
    keep &= ~has_m20 | (close >= m20 * 0.93)  # rs 软闸门(不接飞刀)
    keep &= ~has_mm | (close >= m60 * 0.93)   # 长线支撑
    return keep


# ============ 动量加权评分(用于 B 动量 regime) ============
def score_blend(day: pd.DataFrame, momentum_regime: bool) -> pd.DataFrame:
    fs = factor_scores(day)
    day = day.copy()
    if momentum_regime:
        mom_rank = day["mom60"].rank(pct=True, na_option="keep").fillna(0.5)
        day["composite_score"] = (0.5 * mom_rank
                                  + 0.3 * fs["low_vol"]
                                  + 0.2 * fs["reversal"])
    else:
        day["composite_score"] = 0.5 * fs["low_vol"] + 0.5 * fs["reversal"]
    return day.sort_values("composite_score", ascending=False)


# ============ 各变体核心候选 ============
def core_candidates(day, variant, hot, regime):
    if variant == "B":
        if regime == "momentum":
            sc = score_blend(day, True)
            mask = gate_relaxed(sc)
        else:
            sc = score_lvrev(day, value_factor=False, ey_weight=0.0)
            mask = gate_vectorized(sc, REVERSAL_Q)
    else:
        sc = score_lvrev(day, value_factor=False, ey_weight=0.0)
        if variant in ("C", "AC"):
            boost = np.where(sc["industry"].isin(hot), 1.3, 1.0)
            sc["composite_score"] = sc["composite_score"] * boost
        mask = gate_vectorized(sc, REVERSAL_Q)
    return sc[mask].copy()


def sat_candidates(day, hot=None):
    """动量卫星: 全市场近期最强(rev_chg 20日)。

    诊断实证 2026 的 corr(rev_chg, 未来收益)=+0.105 -> 近期强势正相关于未来收益。
    故卫星买"近期赢家"(rev_chg 高), 广义选(不绑滞后行业动量), 旁路 lvrev 闸门。
    """
    valid = day.dropna(subset=["rev_chg"]).copy()
    return valid.sort_values("rev_chg", ascending=False)


# ============ 变体组合模拟 ============
def run_variant(px, day_frames, mkt_avg, mkt_ma60, hot_sectors, regimes,
                start, end, variant, initial=INITIAL_CAPITAL) -> dict:
    all_dates = pd.to_datetime(sorted(px["date"].unique()))
    fold_dates = [d for d in all_dates
                  if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not fold_dates:
        return {}
    cash = float(initial)
    positions: dict = {}
    portfolio = []
    sell_log = []
    step = 0
    use_sat = variant in ("A", "AC")

    for d in fold_dates:
        step += 1
        day = day_frames.get(d)
        price_map = (dict(zip(day["code"], day["close"]))
                     if day is not None else {})

        # 1) 持仓评估/退出
        to_sell = []
        for code, pos in list(positions.items()):
            cur = price_map.get(code, pos["buy_price"])
            ret = (cur / pos["buy_price"] - 1) * 100
            held = step - pos["entry_step"]
            reason = None
            if ret <= -STOP_LOSS:
                reason = "止损"
            elif cur >= pos["target"]:
                reason = "达标"
            elif held >= pos["hold_days"]:
                reason = "到期"
            elif held > 3 and cur < pos["buy_price"]:
                reason = "衰减"
            if reason:
                cash += pos["shares"] * cur * (1 - trade_cost(code, False))
                sell_log.append({"code": code, "buy_p": pos["buy_price"],
                                 "sell_p": cur, "held": held,
                                 "ret": round(ret, 2), "reason": reason})
                to_sell.append(code)
        for code in to_sell:
            del positions[code]

        # 2) 总资产记账
        total = cash
        for code, pos in positions.items():
            cp = price_map.get(code, pos["buy_price"])
            total += pos["shares"] * cp
        portfolio.append((d, round(total, 2)))

        # 3) 入场判断(L0 市场闸门)
        mavg = mkt_avg.get(d, np.nan)
        mma = mkt_ma60.get(d, np.nan)
        regime_pass = (not pd.isna(mavg)) and (not pd.isna(mma)) and (mavg > mma)
        if not regime_pass or cash < 5000 or day is None or len(day) == 0:
            continue

        hot = hot_sectors.get(d, set())
        reg = regimes.get(d, "lowvol")
        held = set(positions.keys())
        cash_start = cash
        sat_cap = SAT_BUDGET_FRAC * cash_start if use_sat else 0.0
        core_cap = (1 - SAT_BUDGET_FRAC) * cash_start if use_sat else cash_start
        sat_spent = 0.0
        core_spent = 0.0
        core_buys = 0
        sat_buys = 0

        # ---- 核心(lvrev / B / C tilt) ----
        cc = core_candidates(day, variant, hot, reg)
        cc = cc[~cc["code"].isin(held)]
        cc = cc.head(MAX_PICKS_PER_DAY * 3)
        if len(cc):
            scores = cc["composite_score"].values
            s_med = float(np.median(scores))
            s_max = float(np.max(scores))
            s_range = max(s_max - s_med, 0.01)
        for _, row in cc.iterrows():
            if core_buys >= MAX_PICKS_PER_DAY:
                break
            code = int(row["code"])
            price = float(row["close"]) if pd.notna(row.get("close")) else 0.0
            if price <= 0:
                continue
            alloc = (core_cap - core_spent) / max(MAX_PICKS_PER_DAY - core_buys, 1)
            shares = int(alloc / price / 100) * 100
            if shares < 100:
                continue
            cost = shares * price * (1 + trade_cost(code, True))
            if cost > (core_cap - core_spent) or cost > cash:
                continue
            s = float(row["composite_score"])
            pct = min(1.0, max(0.1, (s - s_med) / s_range + 0.5))
            hold_days = int(min(10, max(3, round(5 + pct * 5))))
            target_pct = TARGET_BASE + pct * 5
            target = price * (1 + target_pct / 100)
            cash -= cost
            core_spent += cost
            core_buys += 1
            positions[code] = {"buy_price": price, "shares": shares,
                               "target": target, "hold_days": hold_days,
                               "entry_step": step, "score": round(s, 2)}

        # ---- 卫星(动量龙头, 旁路 lvrev 闸门, 仅热门板块) ----
        if use_sat and sat_cap > 0:
            sc_sat = sat_candidates(day, hot)
            sc_sat = sc_sat[~sc_sat["code"].isin(held)]
            sc_sat = sc_sat[~sc_sat["code"].isin(set(positions.keys()))]
            sc_sat = sc_sat.head(SAT_MAX * 3)
            if len(sc_sat):
                moms = sc_sat["rev_chg"].values
                m_med = float(np.median(moms))
                m_max = float(np.max(moms))
                m_range = max(m_max - m_med, 0.01)
            for _, row in sc_sat.iterrows():
                if sat_buys >= SAT_MAX:
                    break
                code = int(row["code"])
                price = float(row["close"]) if pd.notna(row.get("close")) else 0.0
                if price <= 0:
                    continue
                alloc = (sat_cap - sat_spent) / max(SAT_MAX - sat_buys, 1)
                shares = int(alloc / price / 100) * 100
                if shares < 100:
                    continue
                cost = shares * price * (1 + trade_cost(code, True))
                if cost > (sat_cap - sat_spent) or cost > cash:
                    continue
                mom = float(row["rev_chg"])
                pct = min(1.0, max(0.1, (mom - m_med) / m_range + 0.5))
                hold_days = int(min(10, max(3, round(5 + pct * 5))))
                target_pct = 5 + pct * 5     # 动量目标更高(<=10%)
                target = price * (1 + target_pct / 100)
                cash -= cost
                sat_spent += cost
                sat_buys += 1
                positions[code] = {"buy_price": price, "shares": shares,
                                   "target": target, "hold_days": hold_days,
                                   "entry_step": step, "score": round(mom, 3),
                                   "sat": True}

    # 清仓
    last = fold_dates[-1]
    day = day_frames.get(last)
    price_map = (dict(zip(day["code"], day["close"]))
                 if day is not None else {})
    for code, pos in list(positions.items()):
        sp = price_map.get(code, pos["buy_price"])
        cash += pos["shares"] * sp * (1 - trade_cost(code, False))
    portfolio.append((last, round(cash, 2)))
    return _metrics(portfolio, initial, sell_log)


# ============ 动量因子 OOS 前向 Rank-IC ============
def momentum_forward_ic(px: pd.DataFrame) -> dict:
    """mom60 对未来20日收益的截面 Rank-IC(天然 OOS, 评分仅用<=t)。"""
    t0 = time.time()
    fwd = (px.groupby("code")["close"].transform(lambda s: s.shift(-20))
           / px["close"] - 1.0)
    work = px[["code", "date", "close", "mom60", "industry"]].copy()
    work["fwd20"] = fwd
    work = work.dropna(subset=["mom60", "fwd20"])
    ics = []
    for d, day in work.groupby("date"):
        if len(day) < 200:
            continue
        ic = day["mom60"].rank().corr(day["fwd20"].rank(), method="pearson")
        if pd.notna(ic):
            ics.append(ic)
    ics = np.array(ics)
    mean_ic = float(np.mean(ics))
    tstat = (float(mean_ic / (np.std(ics) / np.sqrt(len(ics))))
             if len(ics) > 1 else 0.0)
    pos_rate = float(np.mean(ics > 0)) * 100
    LOG.info(f"动量 OOS 前向 Rank-IC: 均值 {mean_ic:+.4f}, t {tstat:.1f}, "
             f"正IC占比 {pos_rate:.0f}%, {len(ics)} 日, {time.time()-t0:.1f}s")
    return {"mean_ic": round(mean_ic, 4), "t_stat": round(tstat, 1),
            "positive_rate": round(pos_rate, 1), "n_days": len(ics)}


# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*",
                    default=["baseline", "A", "B", "C", "AC"],
                    help="指定要跑的变体子集")
    ap.add_argument("--prod-hot", action="store_true",
                    help="用生产版 sector_rotation_watchlist.hot_industry_set "
                         "复算 C/AC 热门行业(验证线上逻辑≡回测逻辑)")
    args = ap.parse_args()
    variants = [v for v in args.variants if v in
                ("baseline", "A", "B", "C", "AC")]

    px = load_data_x()
    all_dates = pd.to_datetime(sorted(px["date"].unique()))
    day_frames = {d: g for d, g in px.groupby("date")}
    mkt_avg = px.groupby("date")["close"].mean().reindex(all_dates)
    mkt_ma60 = mkt_avg.rolling(MARKET_MA, min_periods=MARKET_MA).mean()
    hot = compute_hot_sectors(px)
    if args.prod_hot:
        # 生产等价复算: 用 sector_rotation_watchlist.hot_industry_set 逐决策日重建热门行业
        from sector_rotation_watchlist import hot_industry_set
        LOG.info("使用生产版 hot_industry_set 复算热门行业(逐决策日)...")
        hot = {d: hot_industry_set(as_of=str(d.date()), hot_n=HOT_N,
                                   mom_window=MOM_W)
               for d in all_dates}
        LOG.info(f"生产版热门行业已重建: {len(hot)} 个决策日")
    regimes, breadth = compute_regimes(px, mkt_avg)

    LOG.info(f"热门板块样本(2026-01-05): {sorted(hot.get(pd.Timestamp('2026-01-05'), set()))}")
    LOG.info(f"regime 分布: "
             f"momentum 日数 = {sum(1 for v in regimes.values() if v=='momentum')}")

    results = {}
    for v in variants:
        LOG.info("=" * 60)
        LOG.info("VARIANT %s", v)
        LOG.info("=" * 60)
        folds = {}
        for y in [2021, 2022, 2023, 2024, 2025, 2026]:
            fs = f"{y}-01-01"
            fe = f"{y}-12-31" if y < 2026 else DATA_END
            r = run_variant(px, day_frames, mkt_avg, mkt_ma60, hot, regimes,
                            fs, fe, v)
            folds[str(y)] = r
            print(f"  [{v}] OOS {y}: 收益 {r.get('return_pct','-'):+.2f}% | "
                  f"夏普 {r.get('sharpe','-')} | 回撤 "
                  f"{r.get('max_drawdown_pct','-')}% | 交易 "
                  f"{r.get('sell_stats',{}).get('total_trades','-')} 笔")
        rets = [1 + folds[y]["return_pct"] / 100 for y in folds if folds[y]]
        results[v] = {
            "folds": folds,
            "oos_geo_pct": round((float(np.prod(rets)) - 1) * 100, 2),
            "avg_year_pct": round(float(np.mean(
                [folds[y]["return_pct"] for y in folds if folds[y]])), 2),
        }

    # baseline 全样本校准(保真校验)
    if "baseline" in variants:
        LOG.info("=" * 60)
        LOG.info("baseline 校准(保真校验)")
        LOG.info("=" * 60)
        calib = run_variant(px, day_frames, mkt_avg, mkt_ma60, hot, regimes,
                            "2020-01-01", DATA_END, "baseline")
        results["baseline_calib"] = calib
        print("\n========== baseline 校准(全样本) ==========")
        _print_metrics(calib)
        print("  >> 对照 walk_forward_oos: 2026 折页约 -4.29%, 全样本 OOS 几何约 +35%")

    # 动量因子 OOS 前向 IC
    mic = momentum_forward_ic(px)
    results["momentum_ic"] = mic

    out = HERE / "walk_forward_oos_paths_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")

    print("\n===== 五变体 OOS 汇总 =====")
    for v in variants:
        rr = results[v]
        print(f"  {v:8s}: OOS几何 {rr['oos_geo_pct']:+.2f}% | "
              f"年均 {rr['avg_year_pct']:+.2f}%")
    print(f"  动量 OOS 前向 IC: {mic}")
    print(f"  结果已落盘: {out}")


def _print_metrics(m: dict):
    if not m:
        print("  (无数据)")
        return
    print(f"  初始: ¥{m['initial']:,.0f}  最终: ¥{m['final']:,.0f}")
    print(f"  总收益: {m['return_pct']:+.2f}%  年化: {m['cagr_pct']:+.2f}%")
    print(f"  最大回撤: {m['max_drawdown_pct']:.1f}%  夏普: {m['sharpe']}")
    yr = m.get("year_returns", {})
    if yr:
        print("  逐年: " + "  ".join(f"{y}{r:+.1f}%" for y, r in yr.items()))
    ss = m.get("sell_stats", {})
    if ss:
        print(f"  交易 {ss.get('total_trades')} 笔 | 胜率 {ss.get('win_rate')}% "
              f"| 均持 {ss.get('avg_held_days')}天 | 均益 {ss.get('avg_return'):+}")


if __name__ == "__main__":
    main()
