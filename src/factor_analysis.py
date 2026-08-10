"""
factor_analysis.py — 因子绩效分析（借鉴 Qbot alphalens + vnpy 回测指标）

计算每个因子的 IC (Information Coefficient) 和 IR (Information Ratio)，
分析因子预测能力的稳定性。

用法:
    python -m src.factor_analysis
    python -m src.factor_analysis --factor roe,pe,momentum_20d
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import logging

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from database import get_db

logger = logging.getLogger(__name__)


def compute_ic(factor_values: pd.Series, forward_returns: pd.Series) -> dict:
    """
    计算单因子的 IC 统计。
    
    IC = Rank Correlation(factor_scores, forward_returns)
    """
    valid = factor_values.notna() & forward_returns.notna()
    fv = factor_values[valid]
    fr = forward_returns[valid]
    
    if len(fv) < 10:
        return {"ic": 0, "ic_ir": 0, "n": 0}
    
    # Spearman rank IC
    from scipy.stats import spearmanr
    ic, pval = spearmanr(fv, fr)
    
    return {
        "ic": round(ic, 4),
        "ic_pval": round(pval, 4),
        "n": len(fv),
    }


def analyze_factors(days: int = 60) -> pd.DataFrame:
    """
    分析所有因子的 IC 表现。从 factor_scores 和 daily_price 表获取数据。
    """
    db = get_db()
    
    # 获取因子得分记录
    rows = db.conn.execute("""
        SELECT f.run_id, f.code, f.date, f.composite_score,
               f.pe, f.pb, f.roe, f.momentum_20d, f.momentum_60d,
               f.revenue_growth, f.gross_margin
        FROM factor_scores f
        WHERE f.date >= DATE('now', ?)
        ORDER BY f.date DESC
    """, (f'-{days} days',)).fetchall()
    
    if not rows:
        logger.warning(f"近{days}天无因子得分数据，无法分析")
        return pd.DataFrame()
    
    df = pd.DataFrame([dict(r) for r in rows])
    
    # 还需要 T+N 收益来做 IC，这里用简化的截面IC
    # 注：完整的因子分析需要 forward returns，这在当前数据仓库中不存在
    # 这里提供一个因子得分统计摘要
    
    result_rows = []
    factor_cols = ["pe", "pb", "roe", "momentum_20d", "momentum_60d", 
                    "revenue_growth", "gross_margin", "composite_score"]
    
    for col in factor_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        
        result_rows.append({
            "factor": col,
            "mean": round(vals.mean(), 2),
            "std": round(vals.std(), 2),
            "min": round(vals.min(), 2),
            "max": round(vals.max(), 2),
            "count": len(vals),
            "coverage": round(len(vals) / len(df) * 100, 1),
        })
    
    return pd.DataFrame(result_rows)


def generate_report(df: pd.DataFrame) -> str:
    """生成因子分析 Markdown 报告。"""
    lines = [
        f"# 因子绩效分析\n",
        f"> 生成: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 来源: factor_scores 表\n",
    ]
    
    if len(df) == 0:
        lines.append("\n**无数据** — factor_scores 表为空，先运行 `python -m src.daily_brief` 产生数据。\n")
        lines.append("\n> 提示: factor_scores 在每次引擎运行后自动写入。\n")
        return "\n".join(lines)
    
    lines.append("## 因子统计摘要\n")
    lines.append("| 因子 | 均值 | 标准差 | 最小值 | 最大值 | 样本数 | 覆盖率 |")
    lines.append("|------|------|--------|--------|--------|--------|--------|")
    for _, r in df.iterrows():
        lines.append(f"| {r['factor']} | {r['mean']} | {r['std']} | {r['min']} | {r['max']} | {r['count']} | {r['coverage']}% |")
    
    lines.append("\n## 因子相关性矩阵\n")
    # 简单显示是否有高相关因子（可能冗余）
    lines.append("| 说明 |")
    lines.append("|------|")
    lines.append("| 待 IC 数据积累后产出 |")
    
    lines.append(f"\n---\n*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--factors", type=str, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    df = analyze_factors(args.days)
    report = generate_report(df)
    
    save_dir = Path("history") / datetime.now().strftime("%Y-%m-%d")
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"因子分析_{datetime.now().strftime('%m%d')}.md"
    path.write_text(report, encoding="utf-8")
    
    print(f"\n因子分析报告: {path}")
    if len(df) > 0:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
