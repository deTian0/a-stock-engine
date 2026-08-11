"""
pit_fundamentals.py — 时点正确(Point-in-Time)基本面查询

解决回测前视的根因: 不再把"最新截面"套到所有历史交易日, 而是对回测日 T 只暴露
"截至 T 已公告(ann_date <= T)"的财务数据, 以及 T 日当天的估值。

设计
----
  - 财务: fundamentals_pit(code, end_date, ann_date, ...) 存全部报告期 + 公告日。
          lookup 时取 ann_date <= T 中 end_date 最大的一行(最新已公告财报)。
  - 估值: 优先 daily_basic_pit(trade_date = T) 的实际 pe/pb/total_mv;
           若某日未采集, 则退化为 close(T) ÷ PIT eps_ttm/bps 推导(零额外 API, 真 PIT)。
  - 静态: name/industry 来自 fundamentals 快照表(不随时间变, 无前视)。

内存索引
--------
  - 财务: 按 code 分组, 组内按 ann_date 升序排列, bisect 二分定位 -> O(log n)。
  - 估值: daily_basic_pit 透视成 {date: {code: row}}, O(1) 命中。
  - shares: total_mv(最新快照) / close(最新交易日), 用于推导 market_cap 的近似值。

用法
----
    from pit_fundamentals import PITFundamentals
    pit = PITFundamentals()
    rec = pit.financials_as_of("000001", "2021-06-30")   # 该日可用财务
    m   = pit.build_fin_map("2021-06-30", codes)         # 全市场财务截面
    pit.apply_valuation(df, "2021-06-30")                # 原地填 pe/pb/market_cap
"""

import logging
import sys
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

# 确保本目录(src)在 path 中, 支持 `python -m src.pit_fundamentals`
sys.path.insert(0, str(Path(__file__).parent))

from database import get_market_db

logger = logging.getLogger(__name__)


def _norm_date(d) -> int:
    """'2021-06-30' / '20210630' -> int 20210630 (用于比较/二分)。"""
    if d is None:
        return 0
    s = str(d).replace("-", "").strip()
    s = s[:8]
    try:
        return int(s)
    except ValueError:
        return 0


class PITFundamentals:
    """PIT 基本面内存索引与查询。"""

    # 财务字段(时点相关)
    FIN_FIELDS = ["roe", "gross_margin", "debt_ratio",
                  "revenue_growth", "profit_growth", "eps", "eps_ttm", "bps"]

    def __init__(self, db=None):
        self.db = db or get_market_db()
        self._fin_index = {}      # code -> {"ann":[int...], "rows":[{field:val}]}
        self._static = {}         # code -> (name, industry)
        self._shares = {}         # code -> 股本(股), 由最新快照推导
        self._val_by_date = {}    # norm(date) -> {code: {pe,pb,total_mv,...}}
        self._build()

    # ------------------------------------------------------------ 构建
    def _build(self):
        self._build_fin_index()
        self._build_static_and_shares()
        self._build_val_index()

    def _build_fin_index(self):
        """财务按 code 分组, 组内 ann_date 升序, 二分可用。"""
        df = self.db.get_fundamentals_pit_all()
        if len(df) == 0:
            logger.warning("fundamentals_pit 为空 — 先运行 collect_pit_fundamentals")
            return
        df["ann_i"] = df["ann_date"].apply(_norm_date)
        df["end_i"] = df["end_date"].apply(_norm_date)
        # 仅保留有公告日的记录(无公告日无法做 PIT 闸口, 丢弃以防前视)
        df = df[df["ann_i"] > 0].copy()
        grouped = df.groupby("code")
        for code, g in grouped:
            g = g.sort_values("ann_i")
            ann = g["ann_i"].tolist()
            rows = []
            for _, r in g.iterrows():
                rows.append({f: _num(r.get(f)) for f in self.FIN_FIELDS})
            self._fin_index[code] = {"ann": ann, "rows": rows}
        logger.info(f"PIT 财务索引: {len(self._fin_index)} 只, "
                     f"共 {len(df)} 条报告期")

    def _build_static_and_shares(self):
        """name/industry(静态) + 推导股本(用于 market_cap 近似)。"""
        try:
            fdf = self.db.get_fundamentals_table()
        except Exception:
            fdf = pd.DataFrame()
        if len(fdf) == 0:
            return
        # 每股票最新收盘(用于股本推导): 用窗口函数取各 code 自己的最近交易日 close,
        # 避免全局 MAX(date) 当天数据不全(如 2026-08-11 仅 50 行)导致 shares 大量失效。
        # daily_price.code 带 .SZ/.SH/.BJ 后缀, 归一化到 6 位纯数字。
        latest_close = {}
        try:
            rows = self.db.conn.execute("""
                SELECT code, close FROM (
                    SELECT code, close,
                           ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) rn
                    FROM daily_price
                ) WHERE rn = 1
            """).fetchall()
            latest_close = {
                str(r["code"]).replace(".SZ", "").replace(".SH", "")
                .replace(".BJ", "").zfill(6): r["close"]
                for r in rows
            }
        except Exception as e:
            logger.warning(f"取最新收盘失败: {e}")
        for _, r in fdf.iterrows():
            code = str(r.get("code", "")).zfill(6)
            if not code:
                continue
            self._static[code] = (str(r.get("name") or code),
                                  str(r.get("industry") or "其他"))
            mv = _num(r.get("total_mv"))        # 元
            lc = latest_close.get(code)
            if mv and lc and lc > 0:
                self._shares[code] = mv / lc    # 股本(股)
        logger.info(f"静态信息: {len(self._static)} 只, 推导股本 {len(self._shares)} 只")

    def _build_val_index(self):
        """估值透视: {norm(date): {code: row}}。空库则退化为推导估值。"""
        df = self.db.get_daily_basic_pit_all()
        if len(df) == 0:
            logger.info("daily_basic_pit 为空 — 估值将用 close÷PIT财务 推导")
            return
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["d"] = df["trade_date"].apply(_norm_date)
        for d, g in df.groupby("d"):
            self._val_by_date[d] = {
                row["code"]: {
                    "pe": _num(row.get("pe")),
                    "pb": _num(row.get("pb")),
                    "total_mv": _num(row.get("total_mv")),
                }
                for _, row in g.iterrows()
            }
        logger.info(f"PIT 估值索引: {len(self._val_by_date)} 个交易日")

    # ------------------------------------------------------------ 查询
    def financials_as_of(self, code: str, date_str: str) -> dict:
        """返回 code 在 date_str 时点可用的财务记录(ann_date<=date 的最新报告)。

        返回字段: name, industry, roe, gross_margin, debt_ratio,
        revenue_growth, profit_growth, eps_ttm, bps
        """
        code = str(code).zfill(6)
        d = _norm_date(date_str)
        rec = {"name": code, "industry": "其他",
               "roe": np.nan, "gross_margin": np.nan, "debt_ratio": np.nan,
               "revenue_growth": np.nan, "profit_growth": np.nan,
               "eps_ttm": np.nan, "bps": np.nan}
        if code in self._static:
            rec["name"], rec["industry"] = self._static[code]
        idx = self._fin_index.get(code)
        if idx:
            ann = idx["ann"]
            pos = bisect.bisect_right(ann, d) - 1   # 最后一个 ann <= d
            if pos >= 0:
                fin = idx["rows"][pos]
                for f in self.FIN_FIELDS:
                    rec[f] = fin[f]
        return rec

    def build_fin_map(self, date_str: str, codes: list) -> dict:
        """批量构建某日财务截面: {code: financials_as_of 记录}。"""
        return {str(c).zfill(6): self.financials_as_of(c, date_str) for c in codes}

    def apply_valuation(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """原地填充 pe / pb / market_cap(时点正确)。

        优先用 daily_basic_pit 实际值; 缺失则 close(T) ÷ PIT eps_ttm/bps 推导。
        df 必须已有 close 列与 code 列。
        """
        d = _norm_date(date_str)
        val_day = self._val_by_date.get(d)
        codes = df["code"].tolist()
        closes = df["close"].tolist() if "close" in df.columns else [np.nan] * len(codes)

        pe_l, pb_l, mcap_l = [], [], []
        for code, close in zip(codes, closes):
            code = str(code).zfill(6)
            pe = pb = mcap = np.nan
            if val_day and code in val_day:
                vr = val_day[code]
                pe = vr.get("pe") if vr.get("pe") is not None else np.nan
                pb = vr.get("pb") if vr.get("pb") is not None else np.nan
                mcap = vr.get("total_mv") if vr.get("total_mv") is not None else np.nan
            else:
                # 推导估值: 用该日 PIT 财务 + 当日收盘
                # PE = close / 基本EPS(时点正确, 随报告期更新);
                # PB = close / 每股净资产; market_cap = close × 股本(近似)
                fin = self.financials_as_of(code, date_str)
                if close and not np.isnan(close) and close > 0:
                    eps = fin.get("eps")
                    bps = fin.get("bps")
                    if eps and not np.isnan(eps) and eps > 0:
                        pe = close / eps
                    if bps and not np.isnan(bps) and bps > 0:
                        pb = close / bps
                    sh = self._shares.get(code)
                    if sh:
                        mcap = close * sh
            pe_l.append(pe)
            pb_l.append(pb)
            mcap_l.append(mcap)

        df["pe"] = pe_l
        df["pb"] = pb_l
        df["market_cap"] = mcap_l
        return df

    # ------------------------------------------------------------ 诊断
    def diagnostics(self, probe_dates: list = None) -> dict:
        """抽样验证 PIT 生效: 同一只股票在不同历史日取到不同报告期。"""
        if not self._fin_index:
            return {"ok": False, "reason": "财务索引为空"}
        samples = []
        probe_dates = probe_dates or ["2020-06-30", "2022-06-30", "2024-06-30", "2026-06-30"]
        for code in list(self._fin_index.keys())[:5]:
            row = {"code": code, "dates": {}}
            for pd_ in probe_dates:
                rec = self.financials_as_of(code, pd_)
                row["dates"][pd_] = {
                    "roe": None if np.isnan(rec["roe"]) else round(rec["roe"], 2),
                    "profit_growth": None if np.isnan(rec["profit_growth"])
                    else round(rec["profit_growth"], 2),
                }
            samples.append(row)
        return {"ok": True, "samples": samples}


def _num(v):
    """转 float, None/NaN -> nan。"""
    if v is None:
        return np.nan
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan
