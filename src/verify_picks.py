"""
verify_picks.py - T+2 推荐验证工具

验证历史推荐股票在 T+2 交易日的实际涨跌表现。
用于评估选股系统的有效性。

用法:
    python verify_picks.py --date 2026-08-06          # 验证某日推荐
    python verify_picks.py --range 7                   # 验证近7天推荐
    python verify_picks.py --start 2026-07-27 --end 2026-07-29  # 指定区间
"""

import sys
import os
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from westock_cli import get_cli
from local_price_loader import LocalPriceLoader
from database import get_db

logger = logging.getLogger(__name__)


def load_picks_from_brief(brief_path: Path) -> list[dict]:
    """从简报 Markdown 文件中提取推荐的股票代码和名称。"""
    if not brief_path.exists():
        return []

    content = brief_path.read_text(encoding="utf-8")
    picks = []

    # 从表格中提取（质量榜 + 短线榜 + 观察名单）
    in_table = False
    for line in content.split("\n"):
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 3:
                code = cells[1]
                name = cells[2]
                # 简单验证股票代码格式（6位数字）
                if code.isdigit() and len(code) == 6:
                    # 避免重复
                    if code not in [p["code"] for p in picks]:
                        picks.append({"code": code, "name": name})

    return picks


def get_t_plus_2_return(price_loader: LocalPriceLoader, code: str,
                         pick_date: datetime) -> dict:
    """
    计算 T+2 收益率。
    pick_date: 推荐日期
    返回: {code, name, pick_date, t0_close, t2_close, return_pct, status}
    """
    try:
        # 获取足够多的数据
        df = price_loader.get_price(code, days=30)
        if len(df) < 5:
            return {"code": code, "status": "数据不足"}

        df["date"] = pd.to_datetime(df["date"])

        # 找到推荐日当天（或之后最近的交易日）
        pick_date = pd.Timestamp(pick_date)
        mask = df["date"] >= pick_date
        if mask.sum() == 0:
            return {"code": code, "status": "推荐日超出数据范围"}

        t0_idx = mask.idxmax()
        t0_close = float(df.loc[t0_idx, "close"])

        # T+2 = 推荐日后第2个交易日
        if t0_idx + 2 < len(df):
            t2_close = float(df.iloc[t0_idx + 2]["close"])
            ret = (t2_close / t0_close - 1) * 100
            return {
                "code": code,
                "t0_close": t0_close,
                "t2_close": t2_close,
                "return_pct": round(ret, 2),
                "status": "success",
            }
        else:
            return {"code": code, "status": "T+2数据不足（可能未到T+2日）"}

    except Exception as e:
        logger.error(f"验证 {code} 失败: {e}")
        return {"code": code, "status": f"错误: {e}"}


def verify_date(date_str: str, briefs_dir: Path = None) -> dict:
    """验证某一天的推荐。优先从 SQLite 读取，回退到 Markdown 简报。"""
    picks = []

    # 优先从 SQLite 读取
    try:
        db = get_db()
        c = db.conn
        rows = c.execute(
            "SELECT DISTINCT code, name FROM stock_picks WHERE date=?",
            (date_str,)
        ).fetchall()
        if rows:
            picks = [{"code": r["code"], "name": r["name"]} for r in rows]
            logger.info(f"从 SQLite 读取 {date_str} 的推荐: {len(picks)} 只")
    except Exception as e:
        logger.warning(f"从 SQLite 读取失败: {e}")

    # 回退到 Markdown 简报
    if not picks and briefs_dir:
        brief_path = briefs_dir / date_str / "盘前选股简报.md"
        if brief_path.exists():
            picks = load_picks_from_brief(brief_path)

    if not picks:
        return {"date": date_str, "error": "未找到推荐数据", "picks": []}

    price_loader = LocalPriceLoader()
    pick_date = datetime.strptime(date_str, "%Y-%m-%d")

    results = []
    for pick in picks:
        result = get_t_plus_2_return(price_loader, pick["code"], pick_date)
        result["name"] = pick["name"]
        results.append(result)

    return {"date": date_str, "picks": results}


def generate_verification_report(verification_results: list[dict]) -> str:
    """生成验证报告 Markdown。"""
    lines = ["# T+2 推荐验证报告\n"]
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    all_returns = []
    total_picks = 0
    success_count = 0

    for vr in verification_results:
        if "error" in vr:
            lines.append(f"\n## {vr['date']}\n_{vr['error']}_\n")
            continue

        lines.append(f"\n## {vr['date']}（{len(vr['picks'])} 只推荐）\n")
        lines.append("| 代码 | 名称 | T0收盘 | T+2收盘 | T+2涨幅 | 状态 |")
        lines.append("|------|------|--------|---------|---------|------|")

        for r in vr["picks"]:
            total_picks += 1
            if r.get("status") == "success":
                success_count += 1
                ret = r["return_pct"]
                all_returns.append(ret)
                emoji = "🔴" if ret > 0 else "🟢"  # 红涨绿跌（中国习惯）
                lines.append(
                    f"| {r['code']} | {r.get('name', '-')} | "
                    f"{r['t0_close']} | {r['t2_close']} | "
                    f"{emoji} {ret:+.2f}% | ✅ |"
                )
            else:
                lines.append(
                    f"| {r['code']} | {r.get('name', '-')} | "
                    f"- | - | - | {r.get('status', '-')} |"
                )

    # 汇总统计
    lines.append("\n---\n")
    lines.append("## 汇总统计\n")
    lines.append(f"- 总推荐数: {total_picks}")
    lines.append(f"- 成功验证: {success_count}")
    lines.append(f"- 验证成功率: {success_count/total_picks*100:.1f}%" if total_picks > 0 else "- 验证成功率: N/A")

    if all_returns:
        arr = np.array(all_returns)
        positive = np.sum(arr > 0)
        lines.append(f"- 正收益: {positive}/{len(arr)} ({positive/len(arr)*100:.1f}%)")
        lines.append(f"- 平均涨幅: {np.mean(arr):+.2f}%")
        lines.append(f"- 中位数: {np.median(arr):+.2f}%")
        lines.append(f"- 最大涨幅: {np.max(arr):+.2f}%")
        lines.append(f"- 最大跌幅: {np.min(arr):+.2f}%")
        lines.append(f"- 标准差: {np.std(arr):.2f}%")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T+2 推荐验证工具")
    parser.add_argument("--date", help="验证某一天的推荐 (YYYY-MM-DD)")
    parser.add_argument("--range", type=int, help="验证近N天的推荐")
    parser.add_argument("--start", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = Path(__file__).parent.parent / "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    briefs_dir = Path(config["output"]["brief_dir"])

    # 确定验证日期范围
    dates_to_verify = []
    if args.date:
        dates_to_verify = [args.date]
    elif args.range:
        today = datetime.now()
        for i in range(args.range):
            d = today - timedelta(days=i)
            dates_to_verify.append(d.strftime("%Y-%m-%d"))
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        current = start
        while current <= end:
            dates_to_verify.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        print("请指定验证日期: --date / --range / --start + --end")
        sys.exit(1)

    # 执行验证
    results = []
    db = get_db()
    for date_str in dates_to_verify:
        logger.info(f"验证 {date_str}...")
        result = verify_date(date_str, briefs_dir)
        results.append(result)

        # 保存到 SQLite
        if "error" not in result and result["picks"]:
            try:
                db.save_t2_verification(date_str, result["picks"])
                logger.info(f"T+2验证已入库: {date_str}")
            except Exception as e:
                logger.warning(f"T+2验证入库失败 {date_str}: {e}")

    # 生成报告
    report = generate_verification_report(results)
    report_path = Path("briefs") / f"T2验证报告_{datetime.now().strftime('%Y%m%d')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"\n验证报告已生成: {report_path}")
    print(report)


if __name__ == "__main__":
    main()
