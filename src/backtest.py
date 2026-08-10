"""
backtest.py — 多因子选股回测模块（借鉴 Qbot backtrader 模式）

用法:
    python -m src.backtest --start 2026-01-01 --end 2026-08-01
    python -m src.backtest --factor "roe,pe,momentum_20d" --benchmark 000300

输出:
    回测报告: history/回测报告_YYYY-MM-DD.md + .html
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from database import get_db
from database import get_market_db

logger = logging.getLogger(__name__)


def run_backtest(start_date: str, end_date: str, top_n: int = 10,
                 factors: list[str] = None, benchmark: str = "000300") -> dict:
    """
    用历史 stock_picks 数据分析策略表现。

    返回: {return_pct, win_rate, sharpe, max_drawdown, best_pick, worst_pick, ...}
    """
    db = get_db()
    mdb = get_market_db()

    # 获取选股记录
    picks = db.conn.execute("""
        SELECT run_id, DATE(pick_date) as pd, code, name, score
        FROM stock_picks
        WHERE DATE(pick_date) BETWEEN ? AND ?
        ORDER BY pick_date, score DESC
    """, (start_date, end_date)).fetchall()

    if not picks:
        logger.warning(f"回测区间 {start_date}→{end_date} 无选股数据")
        return {"status": "empty", "period": f"{start_date}→{end_date}"}

    # 按日期分组，每天取 top_n
    by_date = defaultdict(list)
    for p in picks:
        by_date[p["pd"]].append(dict(p))

    # 获取每日收盘价（模拟 T+1 买入，T+N 卖出）
    all_codes = list(set(p["code"] for picks_list in by_date.values() for p in picks_list))
    code_list = ",".join(f"'{c}.SZ','{c}.SH'" for c in all_codes)
    prices = mdb.conn.execute(f"""
        SELECT code, date, close FROM daily_price
        WHERE code IN ({code_list})
        ORDER BY code, date
    """).fetchall()

    price_map = defaultdict(dict)
    for r in prices:
        code = r["code"].split(".")[0].zfill(6)
        price_map[code][r["date"]] = r["close"]

    # 模拟交易
    trades = []
    holding_period = 5  # 持有 5 个交易日
    cash = 100000
    positions = {}

    for date_str in sorted(by_date.keys()):
        top_picks = sorted(by_date[date_str], key=lambda x: -x.get("score", 0))[:top_n]

        # 卖出到期的
        expired = [c for c, pos in list(positions.items()) if pos["days"] >= holding_period]
        for code in expired:
            exit_price = _get_price(price_map, code, date_str)
            if exit_price:
                pnl = (exit_price - positions[code]["entry"]) * positions[code]["shares"]
                trades.append({
                    "code": code, "entry_date": positions[code]["date"],
                    "exit_date": date_str, "entry_price": positions[code]["entry"],
                    "exit_price": exit_price, "pnl": pnl,
                    "return_pct": (exit_price / positions[code]["entry"] - 1) * 100,
                })
                cash += exit_price * positions[code]["shares"]
                del positions[code]

        # 买入新推荐（等仓位分配）
        if top_picks and cash > 1000:
            per_stock = min(cash / len(top_picks) * 0.1, 10000)  # 每只最多1万
            for pick in top_picks:
                code = str(pick["code"]).zfill(6)
                if code in positions:
                    continue
                entry_price = _get_price(price_map, code, date_str)
                if entry_price and entry_price > 0:
                    shares = int(per_stock / entry_price / 100) * 100
                    if shares >= 100:
                        positions[code] = {
                            "entry": entry_price, "date": date_str, "shares": shares, "days": 0
                        }
                        cash -= shares * entry_price

        # 持仓天数+1
        for pos in positions.values():
            pos["days"] += 1

    rets = [t["return_pct"] for t in trades if t.get("return_pct") is not None]
    total_pnl = sum(t.get("pnl", 0) for t in trades)

    result = {
        "status": "ok",
        "period": f"{start_date}→{end_date}",
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(sum(1 for r in rets if r > 0) / max(len(rets), 1) * 100, 1),
        "avg_return": round(np.mean(rets), 2) if rets else 0,
        "max_return": round(max(rets), 2) if rets else 0,
        "min_return": round(min(rets), 2) if rets else 0,
        "sharpe": round(np.mean(rets) / max(np.std(rets), 0.01) * np.sqrt(252), 2) if rets else 0,
        "best": max(trades, key=lambda t: t.get("return_pct", -999)) if trades else None,
        "worst": min(trades, key=lambda t: t.get("return_pct", 999)) if trades else None,
    }
    logger.info(f"回测完成: {len(trades)}笔, 胜率{result['win_rate']}%, 总pnl {total_pnl:.0f}")
    return result


def _get_price(price_map, code, date_str):
    """获取某天的收盘价，日期不匹配则取最近的。"""
    prices = price_map.get(code, {})
    if date_str in prices:
        return prices[date_str]
    # 找最近日期
    dates = sorted(prices.keys())
    for d in dates:
        if d >= date_str:
            return prices[d]
    return prices.get(dates[-1]) if dates else None


def generate_report(result: dict) -> str:
    """生成回测报告 Markdown。"""
    lines = [
        f"# 策略回测报告\n",
        f"> 回测区间: {result.get('period','')} | 生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
    ]
    if result["status"] == "empty":
        lines.append("\n**无回测数据** — 该区间内无选股记录。先运行 `python -m src.daily_brief` 产生选股数据。\n")
        return "\n".join(lines)

    lines.append("## 一、核心指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 交易笔数 | {result['total_trades']} |")
    lines.append(f"| 总盈亏 | {result['total_pnl']:+,.0f} 元 |")
    lines.append(f"| 胜率 | {result['win_rate']}% |")
    lines.append(f"| 平均收益 | {result['avg_return']:+.2f}% |")
    lines.append(f"| 最大收益 | {result['max_return']:+.2f}% |")
    lines.append(f"| 最小收益 | {result['min_return']:+.2f}% |")
    lines.append(f"| 年化夏普 | {result['sharpe']} |")
    lines.append("")

    if result.get("best"):
        b = result["best"]
        lines.append(f"\n## 二、最佳交易\n")
        lines.append(f"**{b['code']}**: {b['entry_date']}买@{b['entry_price']} → {b['exit_date']}卖@{b['exit_price']} ({b['return_pct']:+.2f}%)\n")

    if result.get("worst"):
        w = result["worst"]
        lines.append(f"\n## 三、最差交易\n")
        lines.append(f"**{w['code']}**: {w['entry_date']}买@{w['entry_price']} → {w['exit_date']}卖@{w['exit_price']} ({w['return_pct']:+.2f}%)\n")

    lines.append(f"\n---\n*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=(datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--benchmark", default="000300")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(f"回测: {args.start}→{args.end}, top={args.top}")

    result = run_backtest(args.start, args.end, args.top)
    report = generate_report(result)

    save_dir = Path("history") / datetime.now().strftime("%Y-%m-%d")
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"回测报告_{datetime.now().strftime('%m%d')}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\n回测报告: {path}")

    if result["status"] != "empty":
        print(f"  交易笔数: {result['total_trades']}")
        print(f"  胜率: {result['win_rate']}%")
        print(f"  总盈亏: {result['total_pnl']:+,.0f}")
        print(f"  夏普: {result['sharpe']}")


if __name__ == "__main__":
    main()
