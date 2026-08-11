"""
fundamental_store.py — 持久化基本面数据仓库

把「全市场截面基本面」落地为 market.db 的 fundamentals 表（code 主键），
替代原来只存在 market_data_cache 的 7 天临时缓存，使 L4 多因子评分、
回测、报告都能稳定复用同一份基本面快照。

字段覆盖（尽量捞）:
  基础: name, industry, area, list_date, market
  估值(daily_basic): pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, total_mv(元), circ_mv(元)
  财务(fina_indicator, period 过滤):
        roe, roa, gross_margin(毛利率%), debt_ratio(资产负债率%),
        revenue_growth(or_yoy 营收同比%), profit_growth(netprofit_yoy 净利润同比%),
        dedt_net_profit(profit_dedt 扣非净利润绝对值, 元, 仅参考非评分),
        eps, bps, report_period

用法:
    from fundamental_store import FundamentalStore
    store = FundamentalStore()
    store.upsert(df)                 # 写入/更新
    df = store.get_by_codes(codes)   # 读取子集
    df = store.get_all()             # 全表
    store.coverage_report()          # 打印覆盖率
"""

import logging

import pandas as pd
import numpy as np

from database import get_market_db

logger = logging.getLogger(__name__)


class FundamentalStore:
    """fundamentals 表的读写封装（基于 market.db）。"""

    def __init__(self, db=None):
        self.db = db or get_market_db()

    # ---------- 写入 ----------
    def upsert(self, df: pd.DataFrame) -> int:
        """批量 upsert 基本面 DataFrame。返回写入行数。"""
        return self.db.upsert_fundamentals(df)

    # ---------- 读取 ----------
    def get_all(self) -> pd.DataFrame:
        return self.db.get_fundamentals_table()

    def get_by_codes(self, codes: list[str]) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        norm = [str(c).zfill(6) for c in codes]
        return self.db.get_fundamentals_table(norm)

    # ---------- 覆盖率诊断 ----------
    def coverage_report(self) -> dict:
        """统计每列非空覆盖率。返回 {列: (非空数, 占比%)}。"""
        df = self.get_all()
        if len(df) == 0:
            logger.warning("fundamentals 表为空")
            return {}
        total = len(df)
        report = {}
        for col in self.db.FUND_COLUMNS:
            if col not in df.columns:
                report[col] = (0, 0.0)
                continue
            non_null = df[col].notna().sum()
            report[col] = (int(non_null), round(non_null / total * 100, 1))
        # 打印
        logger.info(f"=== fundamentals 覆盖率 (共 {total} 只) ===")
        for col, (n, pct) in report.items():
            logger.info(f"  {col:16s}: {n:5d} ({pct:5.1f}%)")
        return report

    def count(self) -> int:
        return len(self.get_all())


def build_fund_lookup(df: pd.DataFrame) -> dict:
    """
    把 fundamentals DataFrame 转成 L4 评分用的 fund_lookup 字典。
    code -> {pe, pb, roe, gross_margin, debt_ratio, revenue_growth,
             profit_growth, market_cap, name}
    market_cap 由 total_mv 提供（元）。
    """
    lookup = {}
    if df is None or len(df) == 0:
        return lookup
    for _, row in df.iterrows():
        code = str(row.get("code", ""))
        if not code:
            continue
        lookup[code] = {
            "pe": _f(row.get("pe")),
            "pb": _f(row.get("pb")),
            "roe": _f(row.get("roe")),
            "gross_margin": _f(row.get("gross_margin")),
            "debt_ratio": _f(row.get("debt_ratio")),
            "revenue_growth": _f(row.get("revenue_growth")),
            "profit_growth": _f(row.get("profit_growth")),
            "market_cap": _f(row.get("total_mv")),
            "name": str(row.get("name", code)) if pd.notna(row.get("name")) else code,
        }
    return lookup


def _f(v):
    """None/NaN → nan，否则 float。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan
