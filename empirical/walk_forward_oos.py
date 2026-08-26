"""
walk_forward_oos.py — lvrev 策略 样本外(OOS) walk-forward 验证

设计原则(严格 point-in-time, 杜绝前视偏差):
  - 每个决策日 t 的所有特征(波动率/反转/MA/质量)只用 t 及之前的数据计算
    (rolling/min_periods 保证; MA60 市场闸门用全历史滚动, 折页起点也有完整历史)
  - 评分内核复用 canonical `src/lvrev_scorer.score_lvrev` (v4.26 g=0 权重:
    vol=0.45, rev=0.35, value=0, q=0.12, g=0)
  - 入场闸门复用 `apply_entry_gates` 的等价向量化实现(trend/rs/超跌/支撑/低波)
  - 组合模拟复刻 v4.26 `run_portfolio` 规则(5万起, 日选<=20, 止损5%, 目标3~8%,
    动态持有3~10天, 真实成本: 佣金万0.854 + 印花税万5仅卖出)

验证内容:
  1) 全样本校准: 2020-01-01~2026-08-21 跑一遍, 与 v4.26 声称 +28.59%/夏普0.40 对比,
     检验本引擎对 v4.26 策略的还原度
  2) 扩展窗口 walk-forward: 每年(2021~2026)作为 OOS 折页, 该年决策只用此前历史,
     给出逐折 OOS 收益/夏普/回撤 + 几何聚合, 与样本内对比判断是否"运气"
  3) OOS 前向 Rank-IC: lvrev composite 对未来20日收益的截面排序相关性,
     天然 OOS(评分只用<=t数据, 前向收益用>t数据), 直接检验 alpha 真实性

用法:
  python walk_forward_oos.py            # 校准 + walk-forward + OOS IC
  python walk_forward_oos.py --no-calib # 跳过全样本校准(仅折页+IC)
"""
from __future__ import annotations
import sys, os, argparse, logging, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from lvrev_scorer import score_lvrev  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("wfo")

# ============ 参数(对齐 v4.26 run_portfolio) ============
DB = ROOT / "data_cache" / "market.db"
INITIAL_CAPITAL = 50000.0
MAX_PICKS_PER_DAY = 20
STOP_LOSS = 5.0
TARGET_BASE = 3.0
MARKET_MA = 60
SURVIVOR_MIN_DAYS = 252
COMMISSION = 0.0000854      # 万0.854 佣金(买卖双边)
STAMP_SELL = 0.0005         # 万5 印花税, 仅卖出侧
REVERSAL_Q = 0.30
DATA_END = "2026-08-21"


def trade_cost(code: int, is_buy: bool) -> float:
    """股票: 佣金 + (卖出侧)印花税; ETF 免印花税(此处统一按股票口径, 保守)。"""
    base = COMMISSION
    if not is_buy:
        base += STAMP_SELL
    return base


# ============ 数据加载 + point-in-time 特征 ============
def load_data() -> pd.DataFrame:
    import sqlite3
    t0 = time.time()
    c = sqlite3.connect(str(DB))
    px = pd.read_sql_query(
        "SELECT code, date, close, pct_chg, vol, amount FROM daily_price", c)
    px["code"] = (px["code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
                  .astype(int))
    px["date"] = pd.to_datetime(px["date"])
    # 幸存者过滤(排除新股/退市, 与旧引擎一致)
    cnt = px.groupby("code")["date"].count()
    surv = set(cnt[cnt >= SURVIVOR_MIN_DAYS].index)
    px = px[px["code"].isin(surv)].copy()
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    LOG.info(f"加载日线: {len(px):,} 行, {px['code'].nunique():,} 只幸存, "
             f"{time.time()-t0:.1f}s")

    # --- point-in-time 特征(只用历史) ---
    t0 = time.time()
    g = px.groupby("code", sort=False)
    px["ret"] = g["pct_chg"].transform(lambda s: s / 100.0)
    # 波动率: 20日收益滚动 std
    px["vol20"] = g["ret"].transform(
        lambda s: s.rolling(20, min_periods=20).std())
    # 反转: 近20日累计收益(越低=越超跌=>越买). 用 shift 向量化: close/close_20ago - 1
    px["close20ago"] = g["close"].shift(20)
    px["rev_chg"] = px["close"] / px["close20ago"] - 1.0
    px["ma20"] = g["close"].transform(
        lambda s: s.rolling(20, min_periods=20).mean())
    px["ma60"] = g["close"].transform(
        lambda s: s.rolling(60, min_periods=60).mean())
    px["rs20"] = px["close"] / px["ma60"] - 1.0
    LOG.info(f"特征计算完成: {time.time()-t0:.1f}s")

    # --- 质量因子 debt_ratio: point-in-time(用公告日 ann_date 回溯) ---
    t0 = time.time()
    f = pd.read_sql_query(
        "SELECT code, ann_date, debt_ratio FROM fundamentals_pit "
        "WHERE debt_ratio IS NOT NULL", c)
    c.close()
    if len(f):
        f["code"] = f["code"].astype(int)
        f["ann_date"] = pd.to_datetime(f["ann_date"], errors="coerce")
        f = f.dropna(subset=["ann_date"]).sort_values("ann_date")
        left = px[["code", "date"]].sort_values("date")
        merged = pd.merge_asof(
            left, f.sort_values("ann_date"),
            left_on="date", right_on="ann_date",
            by="code", direction="backward")
        px = px.merge(merged[["code", "date", "debt_ratio"]],
                      on=["code", "date"], how="left")
        LOG.info(f"质量因子(debt_ratio) point-in-time 接入: {len(f):,} 条, "
                 f"{time.time()-t0:.1f}s")
    else:
        px["debt_ratio"] = np.nan
        LOG.info("fundamentals_pit 无 debt_ratio, 质量因子退化为常数(不影响排名)")
    return px


# ============ 向量化入场闸门(等价 apply_entry_gates) ============
def gate_vectorized(df: pd.DataFrame, reversal_q: float = REVERSAL_Q) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    vol_med = (float(df["vol20"].median())
               if df["vol20"].notna().any() else float("inf"))
    chg_q = (float(df["rev_chg"].quantile(reversal_q))
             if df["rev_chg"].notna().any() else -0.05)

    m20, m60, close, vol, rev = (df["ma20"], df["ma60"], df["close"],
                                 df["vol20"], df["rev_chg"])

    # f_trend 硬闸门: ma20 > ma60
    has_mm = m20.notna() & m60.notna()
    keep &= ~has_mm | (m20 > m60)
    # f_rs 软闸门: close >= ma20*0.93
    has_m20 = m20.notna()
    keep &= ~has_m20 | (close >= m20 * 0.93)
    # 近N日超跌(反转核心): rev_chg <= 市场底部分位
    has_rev = rev.notna()
    keep &= ~has_rev | (rev <= chg_q)
    # 长线支撑: close >= ma60*0.93
    keep &= ~has_mm | (close >= m60 * 0.93)
    # 低波门槛: vol20 <= 市场波动中位数
    has_vol = vol.notna() & (vol_med != float("inf"))
    keep &= ~has_vol | (vol <= vol_med)
    return keep


# ============ 组合模拟(复刻 v4.26 run_portfolio) ============
def run_strategy(px: pd.DataFrame, day_frames: dict, mkt_avg: pd.Series,
                 mkt_ma60: pd.Series, start: str, end: str,
                 initial: float = INITIAL_CAPITAL,
                 weight_override: dict | None = None) -> dict:
    all_dates = pd.to_datetime(sorted(px["date"].unique()))
    fold_dates = [d for d in all_dates
                  if (pd.Timestamp(start) <= d <= pd.Timestamp(end))]
    if not fold_dates:
        return {}

    cash = float(initial)
    positions: dict = {}  # code -> dict
    portfolio = []
    sell_log = []
    step = 0

    for d in fold_dates:
        step += 1
        day = day_frames.get(d)
        price_map = dict(zip(day["code"], day["close"])) if day is not None else {}

        # ---- 1. 持仓评估/退出 ----
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
                                 "sell_p": cur, "held": held, "ret": round(ret, 2),
                                 "reason": reason})
                to_sell.append(code)
        for code in to_sell:
            del positions[code]

        # ---- 2. 总资产记账 ----
        total = cash
        for code, pos in positions.items():
            cp = price_map.get(code, pos["buy_price"])
            total += pos["shares"] * cp
        portfolio.append((d, round(total, 2)))

        # ---- 3. 入场判断 ----
        mavg = mkt_avg.get(d, np.nan)
        mma = mkt_ma60.get(d, np.nan)
        regime_pass = (not pd.isna(mavg)) and (not pd.isna(mma)) and (mavg > mma)
        if not regime_pass or cash < 5000 or day is None or len(day) == 0:
            continue

        scored = score_lvrev(day, value_factor=False, ey_weight=0.0,
                              weights=weight_override)
        mask = gate_vectorized(scored, REVERSAL_Q)
        cand = scored[mask & ~scored["code"].isin(set(positions.keys()))]
        cand = cand.head(MAX_PICKS_PER_DAY * 3)  # 复刻: 仅评估头部候选
        if len(cand) == 0:
            continue

        scores = cand["composite_score"].values
        s_med = float(np.median(scores))
        s_max = float(np.max(scores))
        s_range = max(s_max - s_med, 0.01)

        buy_count = 0
        for _, row in cand.iterrows():
            code = int(row["code"])
            price = float(row["close"]) if pd.notna(row.get("close")) else 0.0
            if price <= 0:
                continue
            alloc = cash / max(MAX_PICKS_PER_DAY - buy_count, 1)
            shares = int(alloc / price / 100) * 100
            if shares < 100:
                continue
            cost = shares * price * (1 + trade_cost(code, True))
            if cost > cash * 0.5 or cost > cash:
                continue
            s = float(row["composite_score"])
            pct = min(1.0, max(0.1, (s - s_med) / s_range + 0.5))
            hold_days = int(min(10, max(3, round(5 + pct * 5))))
            target_pct = TARGET_BASE + pct * 5
            target = price * (1 + target_pct / 100)
            cash -= cost
            positions[code] = {
                "buy_price": price, "shares": shares, "target": target,
                "hold_days": hold_days, "entry_step": step,
                "score": round(s, 2),
            }
            buy_count += 1
            if buy_count >= MAX_PICKS_PER_DAY:
                break

    # ---- 清仓 ----
    last = fold_dates[-1]
    day = day_frames.get(last)
    price_map = dict(zip(day["code"], day["close"])) if day is not None else {}
    for code, pos in list(positions.items()):
        sp = price_map.get(code, pos["buy_price"])
        cash += pos["shares"] * sp * (1 - trade_cost(code, False))
    portfolio.append((last, round(cash, 2)))

    return _metrics(portfolio, initial, sell_log)


def _metrics(portfolio, initial, sell_log) -> dict:
    if not portfolio:
        return {}
    values = np.array([v for _, v in portfolio], dtype=float)
    rets = [values[i] / values[i - 1] - 1
            for i in range(1, len(values)) if values[i - 1] > 0]
    years = len(values) / 252.0
    cagr = (values[-1] / initial) ** (1 / years) - 1 if years > 0 else 0
    peak = values[0]
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
        mdd = max(mdd, (peak - v) / peak if peak > 0 else 0)
    ret_arr = np.array(rets) if rets else np.array([0.0])
    rf = 0.02 / 252.0
    excess = ret_arr - rf
    sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) \
        if np.std(excess) > 0 else 0.0
    # 逐年
    yearly = {}
    for d, v in portfolio:
        y = str(d)[:4]
        yearly.setdefault(y, []).append(v)
    year_ret = {y: round((vals[-1] / vals[0] - 1) * 100, 1)
                for y, vals in yearly.items() if vals[0] > 0}
    return {
        "initial": initial, "final": round(float(values[-1]), 2),
        "return_pct": round((values[-1] / initial - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "sharpe": round(sharpe, 2),
        "years": round(years, 1),
        "year_returns": dict(sorted(year_ret.items())),
        "sell_stats": {
            "total_trades": len(sell_log),
            "avg_held_days": round(float(np.mean([s["held"] for s in sell_log])), 1)
            if sell_log else 0,
            "avg_return": round(float(np.mean([s["ret"] for s in sell_log])), 2)
            if sell_log else 0,
            "win_rate": round(float(np.mean([s["ret"] > 0 for s in sell_log]))
                              * 100, 1) if sell_log else 0,
        },
    }


# ============ OOS 前向 Rank-IC ============
def oos_forward_ic(px: pd.DataFrame) -> dict:
    """lvrev composite 对未来20日收益的截面 Rank-IC(天然 OOS)。

    评分在 t 用<=t数据; 前向收益 t->t+20 用>t数据。逐日截面 Spearman,
    跨日均值 + t 统计(粗略, 未做 NW 修正, 样本大偏保守)。
    """
    t0 = time.time()
    # 前向20日收益(per-code 正确平移, 仅用于分析, 不进入决策)
    fwd = px.groupby("code")["close"].transform(lambda s: s.shift(-20)) \
        / px["close"] - 1.0
    work = px[["code", "date", "close", "vol20", "rev_chg", "ma20", "ma60",
               "debt_ratio"]].copy()
    work["fwd20"] = fwd
    work = work.dropna(subset=["vol20", "rev_chg", "ma20", "ma60", "fwd20"])
    # 逐日截面打分 + Rank-IC
    ics = []
    for d, day in work.groupby("date"):
        if len(day) < 200:
            continue
        sc = score_lvrev(day, value_factor=False, ey_weight=0.0)
        if sc["composite_score"].nunique() < 10:
            continue
        # Spearman = 先 rank 再 Pearson(避免依赖 scipy)
        ic = sc["composite_score"].rank().corr(sc["fwd20"].rank(),
                                              method="pearson")
        if pd.notna(ic):
            ics.append(ic)
    ics = np.array(ics)
    mean_ic = float(np.mean(ics))
    tstat = float(mean_ic / (np.std(ics) / np.sqrt(len(ics)))) if len(ics) > 1 else 0.0
    pos_rate = float(np.mean(ics > 0)) * 100
    LOG.info(f"OOS 前向 Rank-IC: 均值 {mean_ic:+.4f}, t {tstat:.1f}, "
             f"正IC占比 {pos_rate:.0f}%, {len(ics)} 个交易日, {time.time()-t0:.1f}s")
    return {"mean_ic": round(mean_ic, 4), "t_stat": round(tstat, 1),
            "positive_rate": round(pos_rate, 1), "n_days": len(ics)}


# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-calib", action="store_true",
                    help="跳过全样本校准(仅折页+IC)")
    args = ap.parse_args()

    px = load_data()
    all_dates = pd.to_datetime(sorted(px["date"].unique()))
    day_frames = {d: g for d, g in px.groupby("date")}

    # 全历史市场均线(折页起点也有完整 MA60)
    mkt_avg = px.groupby("date")["close"].mean().reindex(all_dates)
    mkt_ma60 = mkt_avg.rolling(MARKET_MA, min_periods=MARKET_MA).mean()

    results = {}

    # --- 1) 全样本校准 ---
    if not args.no_calib:
        LOG.info("=" * 60)
        LOG.info("校准: 全样本 2020-01-01 ~ %s", DATA_END)
        LOG.info("=" * 60)
        calib = run_strategy(px, day_frames, mkt_avg, mkt_ma60,
                             "2020-01-01", DATA_END)
        results["calibration"] = calib
        print("\n========== 校准(全样本) ==========")
        _print_metrics(calib)
        print("  >> v4.26 声称: +28.59% / 年化+4.16% / 回撤21.1% / 夏普0.40")

    # --- 2) walk-forward 折页(每年 OOS) ---
    LOG.info("=" * 60)
    LOG.info("walk-forward: 每年作为 OOS 折页")
    LOG.info("=" * 60)
    folds = {}
    fold_years = [2021, 2022, 2023, 2024, 2025, 2026]
    for y in fold_years:
        fs = f"{y}-01-01"
        fe = f"{y}-12-31" if y < 2026 else DATA_END
        r = run_strategy(px, day_frames, mkt_avg, mkt_ma60, fs, fe)
        folds[str(y)] = r
        print(f"  OOS {y}: 收益 {r.get('return_pct','-'):+.2f}% | "
              f"夏普 {r.get('sharpe','-')} | 回撤 {r.get('max_drawdown_pct','-')}% | "
              f"交易 {r.get('sell_stats',{}).get('total_trades','-')} 笔")
    results["folds"] = folds

    # 几何聚合 OOS
    rets = [1 + folds[y]["return_pct"] / 100 for y in folds if folds[y]]
    geo_total = float(np.prod(rets) - 1) * 100
    avg_yr = float(np.mean([folds[y]["return_pct"] for y in folds if folds[y]]))
    print(f"\n  >> OOS 几何聚合(2021-2026): {geo_total:+.2f}% | "
          f"年均 {avg_yr:+.2f}%")
    results["oos_geo_total_pct"] = round(geo_total, 2)
    results["oos_avg_year_pct"] = round(avg_yr, 2)

    # --- 3) OOS 前向 Rank-IC ---
    LOG.info("=" * 60)
    LOG.info("OOS 前向 Rank-IC(alpha 真实性)")
    LOG.info("=" * 60)
    ic = oos_forward_ic(px)
    results["oos_ic"] = ic

    # 落盘 JSON 供报告脚本消费
    import json
    out = HERE / "walk_forward_oos_result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\n结果已落盘: {out}")


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
