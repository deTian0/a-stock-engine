"""A/B 驱动: 验证 lvrev 内核接入价值因子(bp/sp) 的增量 alpha。

复用动量缓存(构建一次), 顺序跑两组:
  A) lvrev + L0 闸门 + 价值因子 ON (--value-factor 显式开启, 非默认)
  B) lvrev + L0 闸门 + 无价值因子 OFF (默认配置, 基准)
逐组打印 总收益/年化/回撤/夏普/胜率/笔数 + 逐年(2020-2026)。
两组各生成 HTML 报告; B(默认配置)即 +26.1% 基准。

结论(见 STRATEGY §8.6): 价值因子在 long-only + 低波/反转内核上净拖累 ~10pt,
故代码接入但默认关(VALUE_FACTOR=False), 需 --value-factor 显式开启。
"""
import time
import local_backtest as lb
import yaml

cfg_raw = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}

print("构建缓存(一次, 含真实市值加权收益指数 + PIT 估值 ps_ttm)...", flush=True)
t0 = time.time()
bt = lb.LocalBacktest()
bt.config = cfg_raw
print(f"缓存就绪 {time.time()-t0:.1f}s", flush=True)

lb.ALPHA_MODE = "lowvol_rev"
lb.MARKET_GATE = True

YEARS = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]


def show(tag, pf):
    s = pf["sell_stats"]
    yr = pf.get("year_returns", {})
    mg = pf.get("market_gate", {})
    print(f"\n[{tag}] 总收益{pf['return_pct']:+.1f}% 年化{pf['cagr_pct']:+.1f}% "
          f"最大回撤-{pf['max_drawdown_pct']:.1f}% 夏普{pf['sharpe']} "
          f"胜率{s['win_rate']:.1f}% 笔数{s['total_trades']}")
    print("  逐年:", " | ".join(f"{y}:{yr.get(y,'-'):+.1f}%" for y in YEARS))
    if mg.get("enabled"):
        print(f"  L0: 熊市(清空){mg['bear_days']}天 / 满仓{mg['full_days']} / 轻仓{mg['light_days']} (共{mg['total_days']})")


# A) 价值因子 ON (--value-factor 显式开启, 非默认)
lb.VALUE_FACTOR = True
pf_a = bt.run_portfolio()
show("A) lvrev+L0+价值因子(bp/sp) ON (--value-factor 显式开启)", pf_a)
html_a = lb.generate_html_report(pf_a, cfg_raw)
print("  HTML报告(A, 价值ON):", html_a)

# B) 对照: 价值因子 OFF (默认配置, 基准)
lb.VALUE_FACTOR = False
pf_b = bt.run_portfolio()
show("B) lvrev+L0+无价值因子 OFF (默认配置, 基准)", pf_b)
html_b = lb.generate_html_report(pf_b, cfg_raw)
print("  HTML报告(B, 默认配置):", html_b)

# 增量摘要
print("\n=== 价值因子增量 (A - B) ===")
print(f"  总收益: {pf_a['return_pct']-pf_b['return_pct']:+.1f}pt  "
      f"回撤: {-pf_a['max_drawdown_pct']-(-pf_b['max_drawdown_pct']):+.1f}pt  "
      f"夏普: {pf_a['sharpe']-pf_b['sharpe']:+.2f}")
