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
from guard import setup_protection, teardown_protection, setup_logging

logger = logging.getLogger(__name__)


# —— 纯「推荐买入」口径（T+2 胜率统计, 排除③C观察/③A持仓/⑦追踪）——
# SQLite 过滤用的类目键（与 stock_picks.category 对齐）
BUY_CATEGORIES = ("②A_质量榜", "②B_短线榜", "ETF组合")
# Markdown 简报白名单章节标题关键词（与 daily_brief 章节命名对齐）
BUY_SECTIONS = ("中长线组合", "短线组合", "ETF 组合")
# 章节关键词 -> 类目（仅用于 Markdown 来源标注 category）
_SECTION_TO_CAT = {
    "中长线组合": "②A_质量榜",
    "短线组合": "②B_短线榜",
    "ETF 组合": "ETF组合",
}


def norm_code(code) -> str:
    """标准化 A股/ETF 代码为 6 位字符串（去前导零/后缀、零填充）。

    解决历史库中同标的以 '000006' 与 '6' 两种格式重复存储的问题：
    入库与验证两侧统一走此函数，保证 '6'→'000006' 坍缩为同一键，
    避免同一股票在胜率统计里被重复计数。
    """
    s = str(code).strip()
    s = s.split(".")[0] if "." in s else s
    if s.isdigit():
        s = s.zfill(6)
    return s


def load_picks_from_brief(brief_path: Path) -> list[dict]:
    """从简报 Markdown 提取「推荐买入」候选：仅 ②A中长线 / ②B短线 / ETF 三段。

    采用**白名单**：只有章节标题命中 BUY_SECTIONS 才纳入；其余章节
    （当前持仓、观察名单、持仓追踪等）一律排除，避免非买入标的摊薄胜率分母。
    返回每项带 category 标注（来源为 Markdown 时按章节推断）。

    列定位按**表头列名**（而非固定列序）：简报列序一旦调整（如把代码挪到第三列）
    也不会静默错位 —— cells[1]/cells[2] 硬编码曾是隐患。
    """
    if not brief_path.exists():
        return []

    content = brief_path.read_text(encoding="utf-8")
    picks = []
    in_buy = False
    current_cat = ""
    header_cols: dict = None  # 当前表格: 列名 -> 列索引

    def _find_col(header: dict, names: tuple) -> int | None:
        """优先精确匹配列名，命中第一个返回其索引；无则 None。"""
        for n in names:
            if n in header:
                return header[n]
        return None

    for line in content.split("\n"):
        if line.startswith("## "):
            in_buy = False
            current_cat = ""
            header_cols = None
            for key, cat in _SECTION_TO_CAT.items():
                if key in line:
                    in_buy = True
                    current_cat = cat
                    break
            continue
        if not in_buy:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|")]
            if "---" in line:
                # 分隔行，跳过
                continue
            if header_cols is None:
                # 首个 | 行视为表头，记录 列名->索引（列序无关）
                header_cols = {c: i for i, c in enumerate(cells)}
                continue
            # 数据行：按列名定位代码/名称
            code_idx = _find_col(header_cols, ("代码",))
            name_idx = _find_col(header_cols, ("名称", "股票名称", "证券名称"))
            if code_idx is None or name_idx is None:
                continue
            code = norm_code(cells[code_idx]) if code_idx < len(cells) else ""
            name = cells[name_idx] if name_idx < len(cells) else ""
            # 6 位数字代码（含 ETF，如 515790）；norm_code 已零填充
            if code and code.isdigit() and len(code) == 6:
                if code not in [p["code"] for p in picks]:
                    picks.append({"code": code, "name": name, "category": current_cat})

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
    """验证某一天的「推荐买入」候选。

    口径：仅统计 ②A质量榜 / ②B短线榜 / ETF组合 三段（纯"推荐买入"胜率）。
    数据源取并集：
      1) SQLite stock_picks：过滤 category IN BUY_CATEGORIES（剔除③C/③A/⑦）；
      2) Markdown 简报：白名单三段（ETF 未持久化到 stock_picks，靠简报补足）。
    """
    picks = []
    seen = set()  # key = (规范化code, category)

    def _add(code, name, category):
        nc = norm_code(code)
        if not nc:
            return
        key = (nc, category)
        if key not in seen:
            seen.add(key)
            # 名称规范化：若 name 本身是纯数字（去零副本 '6'），同步为 6 位代码
            nm = nc if (name and str(name).isdigit()) else (name or nc)
            picks.append({"code": nc, "name": nm, "category": category})

    # 1) SQLite 主源：仅买入类目
    try:
        db = get_db()
        c = db.conn
        rows = c.execute(
            "SELECT DISTINCT code, name, category FROM stock_picks "
            "WHERE date=? AND category IN (?,?,?)",
            (date_str, *BUY_CATEGORIES)
        ).fetchall()
        for r in rows:
            _add(r["code"], r["name"], r["category"])
        if rows:
            logger.info(f"SQLite 买入候选 {date_str}: {len(rows)} 只（已过滤非买入类目）")
    except Exception as e:
        logger.warning(f"SQLite 读取失败: {e}")

    # 2) Markdown 简报：白名单三段（含 ETF）
    if briefs_dir:
        brief_path = briefs_dir / date_str / "盘前选股简报.md"
        if brief_path.exists():
            for p in load_picks_from_brief(brief_path):
                _add(p["code"], p["name"], p.get("category", ""))

    if not picks:
        return {"date": date_str,
                "error": "未找到推荐数据（仅统计②A/②B/ETF买入段）",
                "picks": []}

    price_loader = LocalPriceLoader()
    pick_date = datetime.strptime(date_str, "%Y-%m-%d")

    results = []
    for pick in picks:
        result = get_t_plus_2_return(price_loader, pick["code"], pick_date)
        result["name"] = pick["name"]
        result["category"] = pick.get("category", "")
        results.append(result)

    return {"date": date_str, "picks": results}


def generate_verification_report(verification_results: list[dict]) -> str:
    """生成验证报告 Markdown（纯「推荐买入」口径：②A/②B/ETF）。"""
    lines = ["# T+2 推荐验证报告（纯「推荐买入」：②A质量榜 / ②B短线榜 / ETF组合）\n"]
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("> 口径说明：仅统计②A/②B/ETF买入三段，已排除③C观察名单/③A持仓/⑦追踪等非买入标的。\n")

    all_returns = []
    total_picks = 0
    success_count = 0
    cat_returns: dict[str, list] = {}  # category -> [return_pct,...]

    for vr in verification_results:
        if "error" in vr:
            lines.append(f"\n## {vr['date']}\n_{vr['error']}_\n")
            continue

        lines.append(f"\n## {vr['date']}（{len(vr['picks'])} 只买入候选）\n")
        lines.append("| 代码 | 名称 | 类目 | T0收盘 | T+2收盘 | T+2涨幅 | 状态 |")
        lines.append("|------|------|------|--------|---------|---------|------|")

        for r in vr["picks"]:
            total_picks += 1
            cat = r.get("category", "")
            if r.get("status") == "success":
                success_count += 1
                ret = r["return_pct"]
                all_returns.append(ret)
                cat_returns.setdefault(cat, []).append(ret)
                emoji = "🔴" if ret > 0 else "🟢"  # 红涨绿跌（中国习惯）
                lines.append(
                    f"| {r['code']} | {r.get('name', '-')} | {cat} | "
                    f"{r['t0_close']} | {r['t2_close']} | "
                    f"{emoji} {ret:+.2f}% | ✅ |"
                )
            else:
                lines.append(
                    f"| {r['code']} | {r.get('name', '-')} | {cat} | "
                    f"- | - | - | {r.get('status', '-')} |"
                )

    # 汇总统计
    lines.append("\n---\n")
    lines.append("## 汇总统计（纯推荐买入）\n")
    lines.append(f"- 总买入候选数: {total_picks}")
    lines.append(f"- 成功验证: {success_count}")
    lines.append(f"- 数据齐备率: {success_count/total_picks*100:.1f}%（T+2 价格可获取占比，**非胜率**）" if total_picks > 0 else "- 数据齐备率: N/A")

    if all_returns:
        arr = np.array(all_returns)
        positive = int(np.sum(arr > 0))
        lines.append(f"- **正收益（胜率）: {positive}/{len(arr)} ({positive/len(arr)*100:.1f}%)**")
        lines.append(f"- 平均涨幅: {np.mean(arr):+.2f}%")
        lines.append(f"- 中位数: {np.median(arr):+.2f}%")
        lines.append(f"- 最大涨幅: {np.max(arr):+.2f}%")
        lines.append(f"- 最大跌幅: {np.min(arr):+.2f}%")
        lines.append(f"- 标准差: {np.std(arr):.2f}%")

        # 分桶：按类目（②A/②B/ETF）
        if cat_returns:
            lines.append("\n### 按类目分桶（买入胜率）\n")
            lines.append("| 类目 | 样本数 | 胜率 | 平均涨幅 | 中位数 |")
            lines.append("|------|--------|------|----------|--------|")
            for cat, rets in cat_returns.items():
                a = np.array(rets)
                pos = int(np.sum(a > 0))
                label = cat or "未标注"
                lines.append(
                    f"| {label} | {len(a)} | {pos/len(a)*100:.1f}% | "
                    f"{np.mean(a):+.2f}% | {np.median(a):+.2f}% |"
                )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="T+2 推荐验证工具")
    parser.add_argument("--date", help="验证某一天的推荐 (YYYY-MM-DD)")
    parser.add_argument("--range", type=int, help="验证近N天的推荐")
    parser.add_argument("--start", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    setup_protection()

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_file=config.get("logging", {}).get("file", "data_cache/engine.log"),
        level=config.get("logging", {}).get("level", "INFO"),
        console=config.get("logging", {}).get("console", True),
    )

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
        if start > end:
            print("ERROR: --start 必须早于 --end")
            return
        current = start
        while current <= end:
            dates_to_verify.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        print("请指定验证日期: --date / --range / --start + --end")
        return

    try:
        results = []
        db = get_db()
        for date_str in dates_to_verify:
            logger.info(f"验证 {date_str}...")
            result = verify_date(date_str, briefs_dir)
            results.append(result)

            if "error" not in result and result["picks"]:
                try:
                    db.save_t2_verification(date_str, result["picks"])
                except Exception as e:
                    logger.warning(f"T+2验证入库失败 {date_str}: {e}")

        report = generate_verification_report(results)
        report_path = Path("briefs") / f"T2验证报告_{datetime.now().strftime('%Y%m%d')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        print(f"\n验证报告已生成: {report_path}")
        print(report)

    except KeyboardInterrupt:
        logger.warning("用户中断，正在清理...")
    finally:
        teardown_protection()


if __name__ == "__main__":
    main()
