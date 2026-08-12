"""优化网格扫描驱动 — 方向1/2/3 + 用户方向4(周期性波段)。

复用单个 LocalBacktest 实例(缓存/闸门基础只算一次), 顺序跑所有配置,
收集 总收益/年化/最大回撤/夏普/胜率/笔数 + 逐年(2020-2026) + L0熊市天数,
输出汇总表 + CSV(data_cache/scan_results.csv)。

护栏: 默认配置(W=20,Q=0.30,BEAR=0.12,EY=0,VF=False,TG=False,TP=0) 必须复现
+26.1%/-10.4%/351笔(与 CLI `local_backtest.py --skip-tn` 基线一致: 注入 config
使 initial_capital=20000, 与回测现金闸门一致)。否则视为回归, 需先排查再谈优化。
"""
import time
import logging
import csv
from datetime import datetime
from pathlib import Path
import yaml

import local_backtest as lb

logging.getLogger("local_backtest").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

YEARS = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def build_configs():
    """去重后的扫描配置: (tag, W, Q, BEAR, EY, VF, TG, TP)。默认放首位作护栏。"""
    seen = set()
    cfgs = []
    def add(tag, W, Q, BEAR, EY, VF, TG=False, TP=0.0):
        key = (W, Q, BEAR, EY, VF, TG, round(TP, 4))
        if key in seen:
            return
        seen.add(key)
        cfgs.append((tag, W, Q, BEAR, EY, VF, TG, TP))

    # 护栏: 默认配置
    add("DEFAULT", 20, 0.30, 0.12, 0.0, False, False, 0.0)

    # 方向3: 反转入场窗口 × 底部分位
    for W in (10, 20, 25):
        for Q in (0.20, 0.30, 0.40):
            add(f"REV_W{W}_Q{Q}", W, Q, 0.12, 0.0, False)

    # 方向2: L0 闸门熊市回撤阈值
    for B in (0.10, 0.11, 0.12):
        add(f"BEAR_{int(B*100)}", 20, 0.30, B, 0.0, False)

    # 方向1: ey(1/pe_ttm) 加性价值权重
    for EY in (0.05, 0.10, 0.15):
        add(f"EY_{EY}", 20, 0.30, 0.12, EY, False)

    # 方向4(用户): 周期性波段 — 趋势闸门(上涨周期) × 机械止盈(不贪吃)
    for TG in (False, True):
        for TP in (0.0, 0.08, 0.10, 0.12, 0.15):
            if (not TG) and TP == 0.0:
                continue  # 即 DEFAULT, 已在护栏
            tag = f"CYC_TG{int(TG)}_TP{int(round(TP*100))}"
            add(tag, 20, 0.30, 0.12, 0.0, False, TG, TP)

    return cfgs


def run_one(bt, tag, W, Q, BEAR, EY, VF, TG, TP):
    lb.REVERSAL_WINDOW = W
    lb.REVERSAL_Q = Q
    lb.BEAR_DD = BEAR
    lb.EY_WEIGHT = EY
    lb.VALUE_FACTOR = VF
    lb.ALPHA_MODE = "lowvol_rev"
    lb.MARKET_GATE = True
    lb.TREND_GATE = TG
    lb.TAKE_PROFIT = TP
    t0 = time.time()
    pf = bt.run_portfolio()
    s = pf["sell_stats"]
    yr = pf.get("year_returns", {})
    mg = pf.get("market_gate", {})
    return {
        "tag": tag, "W": W, "Q": Q, "BEAR": BEAR, "EY": EY, "VF": VF,
        "TG": TG, "TP": TP,
        "ret": round(pf["return_pct"], 2),
        "cagr": round(pf["cagr_pct"], 2),
        "mdd": round(pf["max_drawdown_pct"], 2),
        "sharpe": round(pf["sharpe"], 3),
        "win": round(s.get("win_rate", 0), 1),
        "trades": s.get("total_trades", 0),
        "bear_days": mg.get("bear_days", 0),
        "years": {y: round(yr.get(y, 0.0), 1) for y in YEARS},
        "sec": round(time.time() - t0, 1),
    }


def main():
    config = load_config()
    print(f"[{datetime.now():%H:%M:%S}] 构造 LocalBacktest(加载缓存+构建PIT)...", flush=True)
    t0 = time.time()
    bt = lb.LocalBacktest()
    bt.config = config   # 注入 config: run_portfolio 用 initial_capital(20000), 复现 +26.1% 护栏
    print(f"  构造完成 {time.time()-t0:.1f}s", flush=True)

    cfgs = build_configs()
    rows = []
    for (tag, W, Q, BEAR, EY, VF, TG, TP) in cfgs:
        try:
            r = run_one(bt, tag, W, Q, BEAR, EY, VF, TG, TP)
        except Exception as e:
            print(f"  [{tag}] 失败: {e}", flush=True)
            continue
        rows.append(r)
        print(f"  [{tag:18s}] 总{r['ret']:+.1f}% 年化{r['cagr']:+.1f}% 回撤-{r['mdd']:.1f}% "
              f"夏普{r['sharpe']} 胜{r['win']:.1f}% 笔{r['trades']} 熊{r['bear_days']}天 "
              f"TG{int(r['TG'])} TP{int(r['TP']*100):>2d} "
              f"| 2022:{r['years']['2022']:+.1f} 2024:{r['years']['2024']:+.1f} 2026:{r['years']['2026']:+.1f} "
              f"({r['sec']}s)", flush=True)

    # 汇总表
    print("\n" + "=" * 110)
    print("扫描汇总 (总收益 / 回撤 / 夏普 / 胜率 / 笔数 / 熊市天 / TG / TP / 逐年2022-2024-2026)")
    print("=" * 110)
    header = f"{'tag':20s} {'ret':>7s} {'mdd':>6s} {'shrp':>5s} {'win':>5s} {'trd':>4s} {'bear':>4s} {'TG':>3s} {'TP':>3s} | 2022  2024  2026"
    print(header)
    for r in rows:
        print(f"{r['tag']:20s} {r['ret']:>+7.1f} {r['mdd']:>6.1f} {r['sharpe']:>5.2f} {r['win']:>5.1f} {r['trades']:>4d} {r['bear_days']:>4d} {int(r['TG']):>3d} {int(r['TP']*100):>3d} | "
              f"{r['years']['2022']:>+5.1f} {r['years']['2024']:>+5.1f} {r['years']['2026']:>+5.1f}")

    # 护栏检查
    default = next((r for r in rows if r["tag"] == "DEFAULT"), None)
    if default:
        print("\n[护栏] DEFAULT 应≈ +26.1% / -10.4% / 351笔")
        ok = (abs(default['ret']-26.1) < 0.5 and abs(default['mdd']-10.4) < 0.5
              and abs(default['trades']-351) < 10)
        print(f"       实测 +{default['ret']:.1f}% / -{default['mdd']:.1f}% / {default['trades']}笔 "
              f"-> {'OK 无回归' if ok else '⚠️ 偏离, 需排查!'}")

    # 写 CSV
    out = "data_cache/scan_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "W", "Q", "BEAR", "EY", "VF", "TG", "TP", "ret", "cagr", "mdd", "sharpe",
                    "win", "trades", "bear_days"] + [f"y{y}" for y in YEARS])
        for r in rows:
            w.writerow([r["tag"], r["W"], r["Q"], r["BEAR"], r["EY"], r["VF"], int(r["TG"]), r["TP"], r["ret"], r["cagr"],
                        r["mdd"], r["sharpe"], r["win"], r["trades"], r["bear_days"]]
                       + [r["years"][y] for y in YEARS])
    print(f"\nCSV -> {out}")
    print(f"[{datetime.now():%H:%M:%S}] 全部完成, 共 {len(rows)} 个配置", flush=True)


if __name__ == "__main__":
    main()
