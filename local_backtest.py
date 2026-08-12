"""
local_backtest.py — 基于本地 SQLite 数据的回测引擎 (CPU 安全版)

特点:
  - 纯本地 SQLite 读取，零网络调用
  - 全串行处理，单文件单日逐一推进
  - 因子评分使用纯 pandas 向量化运算
  - 板块轮动: 每板块 Top5，两周滑动窗口
  - T+N 验证 (1/3/5 日)
  - CPU 保护: 每日处理间隔 sleep(0.2s)

用法:
    python local_backtest.py
"""

import sys, os, time, logging, sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from database import get_market_db
from factor_engine import score_stocks, pick_top_by_sector, filter_candidates
from pit_fundamentals import PITFundamentals

logger = logging.getLogger(__name__)

# === CPU 安全参数 ===
DAY_SLEEP_SEC = 0
BATCH_SLEEP_SEC = 0
MAX_PER_SECTOR = 5

# === 风控参数（v2.2 新增） ===
RISK_STOP_LOSS_PCT = 8.0       # T+1 止损线 (%)
RISK_MAX_SECTOR_PCT = 20.0     # 单板块最大占比 (%)
RISK_MAX_STOCK_PCT = 5.0
T_PERIODS = [1, 3, 5]
MARKET_MA = 60
MARKET_MA200 = 200           # L0 市场闸门: 宽基代理指数 MA200(熊市判定)
MARKET_GATE = True            # L0 市场闸门总开关 (--no-market-gate 关闭做 A/B)
INITIAL_CAPITAL = 50000
MAX_PICKS_PER_DAY = 8        # M4: 每日选股数 20→8 (降低分散度/换手)
TRADE_COST = 0.0013          # 0.13% 交易成本 (股票: 万0.854+印花税)
ETF_COST = 0.00017           # 0.017% ETF交易成本 (免印花税)
STOP_LOSS = 8.0              # 止损 (%) — 2026-08-12 固化最优: 硬止损笔数48.7%→33.2%, 总收益-39.65%→-13.66%
TARGET_BASE = 5.0            # 基础目标收益 (%)  M5: 3→5
TRAIL_STOP_PCT = 6.0         # M2: 移动止损, 自持仓高点回撤比例 (%)
MAX_POSITIONS = 15           # M4: 最大持仓数 (降低分散度)
MAX_HOLD_DAYS = 60            # C1: 动态持有安全上限(极长持有才强制了结, 非正常退出)
MIN_HOLD = 30                 # 最小持有期(天): 非硬止损卖出需持有>=N天 — v4.8 扫描5/10/15/20/25/30, mh30最优(+49.6%, 回撤-27.8%, 夏普0.31, 盈亏比2.48)
PULLBACK_GUARD = False        # 入场回踩不破: 近5日最低价>=MA20 才买(降换手实验)
# 选股 alpha 内核: "trend"(v4.8 买强/动量, 仅作对照) | "lowvol_rev"(低波+反转+质量, 真 alpha 路线, 已固化默认)
# 2026-08-12 固化: lvrev 内核改为"近20日超跌"口径, f_rs/f_trend 降为风控闸门; 叠加 L0 市场闸门(宽基MA200+估值分位)
ALPHA_MODE = "lowvol_rev"


class LocalBacktest:
    """本地数据回测——SQLite Only。"""

    def __init__(self):
        self.db = get_market_db()
        self.raw_conn = sqlite3.connect(self.db.db_path)
        self._dates_cache: Optional[list[str]] = None
        self._survivors: Optional[set] = None  # 存活股票集合
        self._mom_cache: Optional[dict] = None   # M1a: 预计算动量矩阵 {列名: DataFrame(date×code)}
        self._pit: Optional[PITFundamentals] = None  # M6: PIT 时点基本面索引
        self._build_momentum_cache()
        self._build_pit_index()

    def _build_momentum_cache(self):
        """M1a: 一次性从 daily_price 透视 close 矩阵, 预计算各周期动量。

        历史回测原依赖 daily_snapshot_<date> 临时缓存(对历史全空), 导致因子失效、
        退化为纯 pct_chg 排名。这里从日线本身计算, 让动量因子在回测中真正生效。
        """
        logger.info("预计算动量矩阵(close 透视)...")
        t0 = time.time()
        df = pd.read_sql(
            "SELECT code, date, close, pct_chg FROM daily_price", self.raw_conn)
        df["code"] = df["code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
        close = df.pivot(index="date", columns="code", values="close").sort_index()
        chg = df.pivot(index="date", columns="code", values="pct_chg").sort_index()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        self._mom_cache = {
            "chg_3d": close.pct_change(3),
            "chg_6d": close.pct_change(6),
            "chg_10d": close.pct_change(10),
            "chg_20d": close.pct_change(20),   # 近20日超跌(反转核心口径)
            "chg_25d": close.pct_change(25),
            "change_pct": chg,
            "ma20": ma20,   # B1/C1: 相对强度择时需逐股 MA
            "ma60": ma60,
            "min_close_5": close.rolling(5).min(),  # 近5日最低收盘价(回踩不破判定)
            "vol20": chg.rolling(20).std(),  # 20日实现波动率(低波因子, 真 alpha 内核)
        }
        logger.info(f"动量矩阵完成: {close.shape[0]}天×{close.shape[1]}只, 耗时{time.time()-t0:.1f}s")

    def _build_pit_index(self):
        """M6: 构建 PIT 时点基本面索引, 从根上消除回测前视。

        替代原 _build_fund_cache(静态最新截面): 旧逻辑把 2026-08 估值 + 20260331
        财务套到所有历史日, 回测 2020 年却用 2026 年 ROE/PE/市值 -> 教科书级前视。
        现在每个回测日 T 只暴露 ann_date<=T 的最近财报 + T 日估值(见 pit_fundamentals)。
        """
        try:
            self._pit = PITFundamentals()
            logger.info(f"PIT 时点基本面索引构建完成"
                        f"({'有数据' if self._pit._fin_index else '财务为空!'})")
        except Exception as e:
            logger.warning(f"PIT 索引构建失败: {e}")
            self._pit = None

    def _build_market_regime(self):
        """L0 市场闸门: 宽基代理指数 MA200 + 估值分位 联合仓控, 压系统性回撤。

        构造宽基代理指数(全体幸存者收盘价中位数, 抗小盘/上市潮扰动)→ MA200;
        估值分位用 daily_basic_pit 全市场 pe_ttm 中位数在滚动3年窗口的百分位
        (earnings yield 视角, 越高=越便宜)。二者合成每日仓位系数 mkt_scale:
          - 指数 < MA200 (熊市)              → 0.0  清空权益、停止建仓(趋势对冲)
          - 指数 >= MA200 且 昂贵(廉价分位<30%) → 0.5  半仓
          - 指数 >= MA200 且 便宜(廉价分位>70%) → 1.0  满仓
          - 其余(中性)                       → 0.75
        MA200 预热期(<200日)或估值预热不足 → 视为参与(0.75), 不误杀。
        long-only 无空头, 此闸门是压制 2022/2024 系统性回撤的关键(§8.4)。
        """
        import numpy as np
        logger.info("构建 L0 市场闸门(宽基 MA200 + 估值分位)...")
        t0 = time.time()
        c = self.raw_conn
        surv = self._compute_survivors()
        # 统一为 6 位代码, 与 daily_basic_pit(纯6位) / 去后缀后的 daily_price 对齐,
        # 否则带后缀的 surv 与 6 位码 isin 永远不匹配 -> mv 过滤成 0 行 -> 闸门变 no-op (已踩坑)
        surv_s = {code.split(".")[0] for code in surv}

        # 1) 宽基代理指数: 真实市值加权收益指数(非股价水平平均, 后者有成分漂移→假熊市)
        #    每日按个股市值加权收益率复利: level_t = level_{t-1} * (1 + Σ w_i * r_i)
        #    w_i = total_mv_i / Σ total_mv (来自 daily_basic_pit), 与去后缀 daily_price 对齐。
        #    这是中证全指级别"宽基"的可信代理, 用于 L0 系统性回撤闸门。
        dp = pd.read_sql("SELECT date, code, close FROM daily_price", c)
        dp["code"] = dp["code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
        dp = dp[dp["code"].isin(surv_s)]
        mv = pd.read_sql("SELECT trade_date, code, total_mv FROM daily_basic_pit WHERE total_mv > 0", c)
        mv["code"] = mv["code"].astype(str).str.zfill(6)
        mv = mv[mv["code"].isin(surv_s)]
        close_p = dp.pivot(index="date", columns="code", values="close").sort_index()
        mv_p = mv.pivot(index="trade_date", columns="code", values="total_mv").sort_index()
        common = close_p.index.intersection(mv_p.index)
        close_p = close_p.loc[common]
        mv_p = mv_p.reindex(index=common, columns=close_p.columns)
        rets = close_p.pct_change(1)                                 # 个股日收益
        w = mv_p.div(mv_p.sum(axis=1, skipna=True), axis=0).fillna(0)  # 每日市值权重(axis=0广播!)
        idx_ret = (rets * w).sum(axis=1).fillna(0)                   # 指数日收益
        level = (1000.0 * (1.0 + idx_ret).cumprod()).sort_index()    # 指数点位(基准1000)
        idx_ma200 = level.rolling(MARKET_MA200).mean()
        # 自1年高点回撤(战术熊市判据, 比 MA200 更精准, 不误伤震荡/复苏段)
        roll_max = level.rolling(252).max()
        dd = (level / roll_max - 1.0)

        # 2) 估值分位: 全市场 pe_ttm 中位数 → earnings yield 滚动3年百分位
        val = pd.read_sql(
            "SELECT trade_date, pe_ttm FROM daily_basic_pit WHERE pe_ttm > 0", c)
        if len(val):
            val = val.groupby("trade_date")["pe_ttm"].median().sort_index()
            ey = (1.0 / val).replace([np.inf, -np.inf], np.nan)
            win = 756  # ~3 年交易日
            cheap_pct = ey.rolling(win).apply(
                lambda x: float((x[-1] >= x).mean()), raw=True)
        else:
            cheap_pct = pd.Series(dtype=float)

        # 3) 合成每日 mkt_scale — 压系统性回撤(§8.4 核心)
        #    - 指数自1年高点回撤 > 12% (熊市)   → 0.0  清空权益、停止建仓
        #    - 非熊市 且 市场极贵(廉价分位<20%) → 0.6  轻仓
        #    - 其余(正常/便宜)                    → 1.0  满仓
        #    MA200 仅作统计展示, 不单独触发清仓(避免2023震荡段被过度清空)
        BEAR_DD = 0.12
        scale = {}
        for d in level.index:
            dd_v = dd.get(d, np.nan)
            if (not pd.isna(dd_v)) and dd_v <= -BEAR_DD:
                scale[d] = 0.0          # 熊市(自1年高点回撤>12%) → 清空
            else:
                cp_v = cheap_pct.get(d, np.nan)
                scale[d] = 0.6 if (not pd.isna(cp_v)) and cp_v < 0.20 else 1.0

        self._mkt_scale = scale
        self._mkt_ma200 = idx_ma200.to_dict()
        self._mkt_cheap = cheap_pct.to_dict()
        self._mkt_index = level.to_dict()
        self._mkt_dd = dd.to_dict()
        logger.info(f"L0 市场闸门就绪: 指数末值{level.iloc[-1]:.0f}, "
                    f"熊市(回撤>{int(BEAR_DD*100)}%) {sum(1 for v in scale.values() if v<=0)}天, "
                    f"耗时{time.time()-t0:.0f}s")
        bear_days = sum(1 for v in scale.values() if v <= 0)
        logger.info(f"L0 闸门完成: {bear_days}/{len(scale)} 天判定熊市(指数<MA200), 耗时{time.time()-t0:.1f}s")

    def _compute_survivors(self, min_days: int = 252) -> set:
        """幸存者偏差校正：排除上市<1年的新股和即将退市的。"""
        if self._survivors is not None:
            return self._survivors
        c = self.raw_conn
        rows = c.execute(
            "SELECT code, COUNT(*) as cnt FROM daily_price GROUP BY code HAVING cnt >= ?",
            (min_days,)
        ).fetchall()
        self._survivors = {r[0] for r in rows}
        logger.info(f"幸存者过滤: {len(self._survivors)} 只 (>={min_days}天)")
        return self._survivors  # 缓存最近读取的日期数据

    def get_available_dates(self) -> list[str]:
        """获取所有可用交易日（后复权CSV + parquet 合并）。"""
        if self._dates_cache is not None:
            return self._dates_cache
        c = self.raw_conn
        rows = c.execute("SELECT DISTINCT date FROM daily_price ORDER BY date").fetchall()
        self._dates_cache = [r[0] for r in rows if r[0]]
        logger.info(f"可用日期: {len(self._dates_cache)} 天 ({self._dates_cache[0]} ~ {self._dates_cache[-1]})")
        return self._dates_cache

    def load_day_data(self, date_str: str) -> pd.DataFrame:
        """加载截面：parquet价格 + 后复权基本面合并（无缓存）。"""
        # 从 daily_price 读价格
        c = self.raw_conn
        rows = c.execute(
            "SELECT code, close, pct_chg, vol, amount FROM daily_price WHERE date=?",
            (date_str,)
        ).fetchall()
        if not rows:
            return pd.DataFrame()

        # 幸存者过滤
        survivors = self._compute_survivors()
        rows = [r for r in rows if r[0] in survivors]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["code_raw", "close", "pct_chg", "vol", "amount"])
        # 清理后缀用于匹配基本面
        df["code"] = df["code_raw"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
        df["sector"] = "其他"
        df["name"] = df["code"]

        # === M1a: 合并动量(从预计算矩阵, 历史回测真正生效) ===
        if self._mom_cache:
            for col in ["chg_3d", "chg_6d", "chg_10d", "chg_20d", "chg_25d", "change_pct", "vol20"]:
                frame = self._mom_cache.get(col)
                if frame is not None and date_str in frame.index:
                    df[col] = df["code"].map(frame.loc[date_str])

        # === B1/C1: 合并相对强度(价格/MA-1)与趋势(MA20>MA60) ===
        # 供 factor_engine 的 f_rs / f_trend 因子消费, 并实现"只买站上均线的强势股"。
        ma20_f = self._mom_cache.get("ma20")
        ma60_f = self._mom_cache.get("ma60")
        if ma20_f is not None and date_str in ma20_f.index:
            df["ma20"] = df["code"].map(ma20_f.loc[date_str])
            df["ma60"] = df["code"].map(ma60_f.loc[date_str])
            # 防御: close>0 且 ma>0 才算, 否则 NaN
            df["rs20"] = np.where((df["close"] > 0) & (df["ma20"] > 0),
                                  (df["close"] / df["ma20"] - 1) * 100, np.nan)
            df["rs60"] = np.where((df["close"] > 0) & (df["ma60"] > 0),
                                  (df["close"] / df["ma60"] - 1) * 100, np.nan)
            df["trend_up"] = (df["ma20"] > df["ma60"])
            # 回踩不破(降换手实验): 近5日最低价>=MA20(留2%容差防噪音), 确认站稳而非刚破即买
            minc5_f = self._mom_cache.get("min_close_5")
            if minc5_f is not None and date_str in minc5_f.index:
                df["min_close_5"] = df["code"].map(minc5_f.loc[date_str])
                df["pullback_ok"] = np.where(
                    (df["min_close_5"].notna()) & (df["ma20"] > 0),
                    df["min_close_5"] >= df["ma20"] * 0.98, True)
            else:
                df["pullback_ok"] = True

        # === M6: 合并 PIT 时点基本面(ann_date<=T 的最近财报 + T 日估值) ===
        if self._pit is not None:
            codes = df["code"].tolist()
            fin_map = self._pit.build_fin_map(date_str, codes)
            # 财务 + 静态 name/industry
            df["name"] = df["code"].map(lambda c: fin_map.get(str(c).zfill(6), {}).get("name", c))
            df["sector"] = df["code"].map(
                lambda c: fin_map.get(str(c).zfill(6), {}).get("industry", "其他"))
            for col in ["roe", "gross_margin", "debt_ratio",
                        "revenue_growth", "profit_growth", "eps_ttm", "bps"]:
                df[col] = df["code"].map(lambda c: fin_map.get(str(c).zfill(6), {}).get(col))
            # 估值(实际 daily_basic_pit 或 close÷PIT财务 推导) — 真正 PIT
            self._pit.apply_valuation(df, date_str)

        return df

    def filter_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """L2 过滤：有基本面 → factor_engine，纯价格 → 简化过滤。"""
        has_fundamentals = "pe" in df.columns and df["pe"].notna().sum() > 100
        if has_fundamentals:
            return filter_candidates(df)
        # 纯价格过滤：排除价格异常
        c = df.copy()
        if "close" in c.columns:
            c = c[c["close"] > 0]
        if "pct_chg" in c.columns:
            c = c[c["pct_chg"] < 20]  # 排除涨停
        return c

    def score_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """评分：有基本面→factor_engine，纯价格→动量因子。"""
        if ALPHA_MODE == "lowvol_rev":
            return self._score_lowvol_rev(df)
        has_fundamentals = "market_cap" in df.columns and df["market_cap"].notna().sum() > 100
        if has_fundamentals:
            # M3+: 权重可由 config.backtest.factor_weights 覆盖, 便于调参
            weights = None
            if getattr(self, "config", None):
                weights = self.config.get("backtest", {}).get("factor_weights")
            return score_stocks(df, weights=weights)
        # 纯动量评分
        s = df.copy()
        if "pct_chg" in s.columns:
            s["composite_score"] = s["pct_chg"].fillna(0)
        else:
            s["composite_score"] = 0
        return s.sort_values("composite_score", ascending=False)

    def _score_lowvol_rev(self, df: pd.DataFrame) -> pd.DataFrame:
        """真 alpha 内核: 低波动(正) + 反转(近20日超跌) + 质量 + 成长。

        依据 alpha_research.py 诊断:
          - 低波动是唯一稳健正 alpha (IC+0.11, 多空+23%/yr)
          - 趋势/动量/RS 为反 alpha (IC 显著负) -> 改用「买近20日超跌」反转
          - 质量/成长 ≈ 零 alpha, 仅作风险稳定器
        权重: 低波0.45 / 反转0.35 / 质量0.12 / 成长0.08
        入场口径见 run_portfolio: f_trend(MA20>MA60)硬闸门 + f_rs(相对强度)软闸门(不接飞刀)
          + 近20日超跌(底部30%分位)核心判据, 而非"买强"。
        """
        d = df.copy()
        n = len(d)

        # 1) 低波动: 波动率越低越好 -> (1 - 排名)
        if "vol20" in d.columns:
            s_vol = d["vol20"].rank(pct=True, na_option="keep")
        else:
            s_vol = pd.Series(0.5, index=d.index)
        low_vol = (1 - s_vol.fillna(0.5))

        # 2) 反转: 近20日越超跌越买(均值回归) — "近20日超跌"口径; IC 显著负即说明买弱胜买强
        #    chg_20d 是 alpha_research 中反 alpha 最强的动量窗口(IC20=-0.086), 反向即最强反转信号
        if "chg_20d" in d.columns:
            s_rev = (-d["chg_20d"]).rank(pct=True, na_option="keep")
        elif "chg_10d" in d.columns:
            s_rev = (-d["chg_10d"]).rank(pct=True, na_option="keep")
        else:
            s_rev = pd.Series(0.5, index=d.index)
        reversal = s_rev.fillna(0.5)

        # 3) 质量: ROE 越高越好, 负债率越低越好
        s_q = pd.Series(0.5, index=d.index)
        if "roe" in d.columns:
            s_q = s_q + d["roe"].clip(-20, 40).fillna(0) / 40.0
        if "debt_ratio" in d.columns:
            s_q = s_q - (d["debt_ratio"].clip(0, 100).fillna(50) / 100.0)
        s_q = s_q.rank(pct=True, na_option="keep").fillna(0.5)
        quality = s_q

        # 4) 成长: 营收增速越高越好(零 alpha, 稳定器)
        if "revenue_growth" in d.columns:
            s_g = d["revenue_growth"].clip(-50, 100).fillna(0).rank(pct=True, na_option="keep")
        else:
            s_g = pd.Series(0.5, index=d.index)
        growth = s_g.fillna(0.5)

        w = dict(vol=0.45, rev=0.35, q=0.12, g=0.08)
        d["composite_score"] = (
            w["vol"] * low_vol + w["rev"] * reversal +
            w["q"] * quality + w["g"] * growth
        )
        return d.sort_values("composite_score", ascending=False)

    def pick_by_sector(self, df: pd.DataFrame, max_per: int = 5) -> list[dict]:
        """按板块分组选 Top N（委托 factor_engine）。"""
        return pick_top_by_sector(df, max_per)

    @staticmethod
    def _zscore_within_group(df, col, direction=1):
        return pd.Series(0, index=df.index)  # no longer used, kept for compat

    def pick_by_sector(self, df: pd.DataFrame, max_per: int = 5) -> list[dict]:
        """按板块分组选 Top N（委托 factor_engine）。"""
        return pick_top_by_sector(df, max_per)

    def apply_risk_controls(self, picks: list[dict]) -> list[dict]:
        """风控过滤（v2.2）：板块集中度 + 单股权重限制。"""
        if not picks:
            return picks
        total = len(picks)
        max_per_sector = int(total * RISK_MAX_SECTOR_PCT / 100)

        # 统计板块计数
        sector_counts = {}
        filtered = []
        for p in picks:
            s = p.get("sector", "未知")
            cnt = sector_counts.get(s, 0)
            if cnt >= max_per_sector:
                continue  # 板块超限，跳过
            if cnt >= total * RISK_MAX_STOCK_PCT / 100 * 10:
                continue  # 单股超限
            sector_counts[s] = cnt + 1
            filtered.append(p)

        return filtered

    def get_price_on_date(self, code: str, date_str: str) -> Optional[float]:
        """查询股票收盘价（自动补后缀）。"""
        c = self.raw_conn
        # 尝试无后缀、.SZ、.SH、.BJ
        for suffix in ["", ".SZ", ".SH", ".BJ"]:
            lookup = code + suffix
            row = c.execute(
                "SELECT close FROM daily_price WHERE code=? AND date=?",
                (lookup, date_str)
            ).fetchone()
            if row:
                return row[0]
        return None

    def get_ma_on_date(self, code: str, date_str: str, window: int) -> Optional[float]:
        """查询个股在指定日期的均线值(从预计算矩阵), 供相对强度择时/动态持有判定。"""
        frame = self._mom_cache.get(f"ma{window}")
        if frame is None or date_str not in frame.index:
            return None
        try:
            v = frame.at[date_str, code]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    def verify_performance(self, picks: list[dict], entry_date: str, periods: list[int] = None) -> list[dict]:
        """T+N 收益验证（直查 SQL，不加载全天数据）。"""
        if periods is None:
            periods = [1, 3, 5]

        all_dates = self.get_available_dates()
        date_idx = None
        for i, d in enumerate(all_dates):
            if d >= entry_date:
                date_idx = i
                break
        if date_idx is None:
            return [dict(p, **{f"T+{p}_ret": None for p in periods}) for p in picks]

        results = []
        for pick in picks:
            code = pick["code"]
            entry_price = pick["close"]
            perf = dict(pick)

            for p in periods:
                target_idx = date_idx + p
                if target_idx >= len(all_dates):
                    perf[f"T+{p}_ret"] = None
                    continue

                exit_price = self.get_price_on_date(code, all_dates[target_idx])
                if exit_price and entry_price and entry_price > 0:
                    perf[f"T+{p}_ret"] = round((exit_price / entry_price - 1) * 100, 2)
                else:
                    perf[f"T+{p}_ret"] = None

            results.append(perf)

        return results

    def run(self, start_date: str = None, end_date: str = None) -> dict:
        """运行回测（纯计算，不写DB）。"""
        all_dates = self.get_available_dates()
        keep_last = T_PERIODS[-1] + 5  # 只保留 T+N 验证所需天数

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        logger.info(f"回测区间: {all_dates[0]} ~ {all_dates[-1]}, {len(all_dates)} 天")

        # === 预计算市场择时信号 (MA60) ===
        logger.info(f"计算择时信号 (MA{MARKET_MA})...")
        c = self.raw_conn
        market_avg = {}
        for i, d in enumerate(all_dates):
            r = c.execute("SELECT AVG(close) FROM daily_price WHERE date=?", (d,)).fetchone()
            market_avg[d] = r[0] if r[0] else 0
            if (i + 1) % 500 == 0:
                logger.info(f"  均价: {i+1}/{len(all_dates)}")

        avg_vals = [market_avg[d] for d in all_dates]
        ma60 = [sum(avg_vals[max(0, i - MARKET_MA + 1):i + 1]) / min(i + 1, MARKET_MA)
                for i in range(len(all_dates))]
        market_regime = {all_dates[i]: avg_vals[i] > ma60[i] for i in range(len(all_dates))}
        trade_days = sum(market_regime.values())
        logger.info(f"择时: {trade_days}/{len(all_dates)} 天可交易 ({trade_days/len(all_dates)*100:.0f}%)")

        daily_records = []
        total_picks = 0

        # 缓存本轮选股，批量写入 DB

        # 边跑边验：当天选股如果 T+N 数据已就绪，即时验证
        verify_results = []
        verify_risk = []
        max_period = max(T_PERIODS)
        date_index = {d: i for i, d in enumerate(all_dates)}  # O(1) lookup

        for di, date_str in enumerate(all_dates):
            if (di + 1) % 100 == 0:
                logger.info(f"  进度: {di+1}/{len(all_dates)}")

            # 择时：市场在 MA60 以下 → 空仓
            if not market_regime.get(date_str, True):
                continue

            df = self.load_day_data(date_str)
            if len(df) == 0:
                continue

            df = self.filter_stocks(df)
            df = self.score_stocks(df)
            picks = self.pick_by_sector(df, MAX_PER_SECTOR)
            picks_risk = self.apply_risk_controls(picks)

            if picks:
                daily_records.append({
                    "date": date_str, "picks": picks, "picks_risk": picks_risk,
                    "filtered": len(df)
                })
                total_picks += len(picks)

            # 即时验证：往前找最早可验证的日期（T+N数据已就绪）
            for rec in list(daily_records):  # 遍历副本
                if date_str >= rec["date"]:  # 还没到未来，跳过
                    idx_then = date_index[rec["date"]]
                    if di - idx_then >= max_period:
                        v = self.verify_performance(rec["picks"], rec["date"])
                        verify_results.extend(v)
                        for item in v:
                            item_risk = dict(item)
                            t1 = item_risk.get("T+1_ret")
                            if t1 is not None and t1 < -RISK_STOP_LOSS_PCT:
                                item_risk["T+1_ret"] = -RISK_STOP_LOSS_PCT
                            verify_risk.append(item_risk)
                        daily_records.remove(rec)  # 已验，移除

        logger.info(f"回测完成: {len(daily_records)} 个未验日, {total_picks} 次推荐, {len(verify_results)} 次验证")

        stats = self._compute_stats(verify_results)
        stats_risk = self._compute_stats(verify_risk)

        # 风控统计摘要
        raw_picks = sum(len(r["picks"]) for r in daily_records)
        risk_picks = sum(len(r.get("picks_risk", r["picks"])) for r in daily_records)

        return {
            "total_dates": len(all_dates),
            "valid_dates": len(daily_records),
            "total_picks": total_picks,
            "stats": stats,
            "stats_risk": stats_risk,
            "risk_summary": {
                "before_risk": raw_picks,
                "after_risk": risk_picks,
                "removed": raw_picks - risk_picks,
                "stop_loss": RISK_STOP_LOSS_PCT,
                "max_sector_pct": RISK_MAX_SECTOR_PCT,
                "max_stock_pct": RISK_MAX_STOCK_PCT,
            },
            "verify_results": verify_results,
        }

    def _compute_stats(self, verify_results: list) -> dict:
        """计算统计指标。"""
        all_ret = []
        for v in verify_results:
            for key in ["T+1_ret", "T+3_ret", "T+5_ret"]:
                if key in v and v[key] is not None:
                    all_ret.append({"key": key, "return": v[key]})

        if not all_ret:
            return {}

        df_ret = pd.DataFrame(all_ret)
        stats = {}
        for period in ["T+1_ret", "T+3_ret", "T+5_ret"]:
            vals = [r["return"] for r in all_ret if r["key"] == period]
            if vals:
                arr = np.array(vals)
                stats[period] = {
                    "count": len(arr),
                    "avg": round(float(np.mean(arr)), 2),
                    "win_rate": round(float(np.sum(arr > 0)) / len(arr) * 100, 1),
                    "max": round(float(np.max(arr)), 2),
                    "min": round(float(np.min(arr)), 2),
                }

        # 高频推荐
        code_freq = {}
        for v in verify_results:
            code = v.get("code", "")
            code_freq[code] = code_freq.get(code, 0) + 1
        top_codes = sorted(code_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        stats["most_picked"] = [
            {"code": c, "name": "", "hits": f} for c, f in top_codes
        ]

        return stats


    @staticmethod
    def _trade_cost(code: str, is_buy: bool = True) -> float:
        """计算交易成本率。ETF 免印花税。"""
        if code.startswith(("15", "51", "56", "58", "510", "512", "513", "515", "516",
                            "517", "518", "560", "561", "562", "563", "588")):
            return ETF_COST  # ETF 只有佣金，无印花税
        return TRADE_COST

    def run_portfolio(self) -> dict:
        """资金模拟：每只有目标/止损，动态持仓周期。初始资金读配置 portfolio.initial_capital。"""
        all_dates = self.get_available_dates()
        all_dates = [d for d in all_dates if d >= "2020-01-01"]  # 只回测 2020+
        c = self.raw_conn

        # 初始资金：优先使用配置 portfolio.initial_capital（与实盘 available_cash 对齐）
        cap = INITIAL_CAPITAL
        if getattr(self, "config", None):
            cap = float(self.config.get("portfolio", {}).get("initial_capital", INITIAL_CAPITAL))

        # 市场择时
        market_avg = {}
        for d in all_dates:
            r = c.execute("SELECT AVG(close) FROM daily_price WHERE date=?", (d,)).fetchone()
            market_avg[d] = r[0] or 0
        avg_vals = [market_avg[d] for d in all_dates]
        ma60 = [sum(avg_vals[max(0,i-MARKET_MA+1):i+1])/min(i+1,MARKET_MA) for i in range(len(all_dates))]
        regime = {all_dates[i]: avg_vals[i] > ma60[i] for i in range(len(all_dates))}

        # L0 市场闸门(宽基MA200 + 估值分位) — 压系统性回撤(§8.4)
        if MARKET_GATE:
            self._build_market_regime()
            mkt_scale = self._mkt_scale
        else:
            mkt_scale = {d: 1.0 for d in all_dates}  # --no-market-gate: 满仓对照

        cash = float(cap)
        positions = {}  # {code: {buy_price, shares, target, stop, hold_days, entry_idx, score}}
        portfolio = []
        sell_log = []   # 记录每笔卖出: (code, buy_p, sell_p, held, ret%, reason)

        logger.info(f"资金模拟: 初始{cap:,.0f}元, 日选{MAX_PICKS_PER_DAY}只")
        logger.info(f"规则: 目标+{TARGET_BASE}~8%, 止损-{STOP_LOSS}%, 成本{TRADE_COST*100:.1f}%")

        for di, date_str in enumerate(all_dates):
            if (di + 1) % 200 == 0:
                logger.info(f"  进度: {di+1}/{len(all_dates)} | 持仓{len(positions)} | 现金{cash:,.0f}")

            # === 每日持仓评估 (C1: 动态持有决策, 取消固定到期) ===
            # L0 市场闸门(每日): 熊市(广基指数自1年高点回撤>12%) → bear=True, 强制清仓且不新建仓
            mscale = mkt_scale.get(date_str, 1.0)
            bear = (mscale <= 0.0)

            to_sell = []
            for code, pos in list(positions.items()):
                cur_price = self.get_price_on_date(code, date_str)
                if cur_price is None:
                    cur_price = pos["buy_price"]
                # 更新持仓高点
                if cur_price > pos.get("peak", pos["buy_price"]):
                    pos["peak"] = cur_price
                held_days = di - pos["entry_idx"]
                ret_pct = (cur_price / pos["buy_price"] - 1) * 100
                peak_ret = (pos["peak"] / pos["buy_price"] - 1) * 100

                # 相对强度择时判定 (B1/C1): 个股均线位置
                ma20 = self.get_ma_on_date(code, date_str, 20)
                ma60 = self.get_ma_on_date(code, date_str, 60)
                trend_up = (ma20 is not None and ma60 is not None and ma20 > ma60)
                below_ma20 = (ma20 is not None and cur_price < ma20)

                reason = None
                cr = self._trade_cost(code)
                # 0. L0 市场熊市闸门(绕过MIN_HOLD): 广基指数自1年高点回撤>12% 强制清仓, 压系统性回撤
                if bear:
                    reason = "市场熊市清仓(回撤>12%)"
                # 1. 硬止损 (始终允许, 保命, 不受最小持有期约束)
                elif ret_pct <= -STOP_LOSS:
                    reason = f"止损({ret_pct:+.1f}%)"
                # 其余主动卖出需满足最小持有期(降换手实验: 避免短线频繁进出)
                elif held_days >= MIN_HOLD:
                    # 2. 达到目标(止盈)
                    if cur_price >= pos["target"]:
                        reason = f"达到目标({ret_pct:+.1f}%)"
                    # 3. 动态放弃(C1): 趋势破位(MA20 下穿 MA60) → 不再持有
                    elif ma20 is not None and ma60 is not None and not trend_up:
                        reason = f"趋势破位({ret_pct:+.1f}%)"
                    # 4. 动态放弃(C1): 跌破 MA20 且已持有≥2天(留1天缓冲防噪音) → 放弃
                    #    lowvol_rev 模式: 入场即在 MA20 附近/下方, 此规则会首日即卖, 故跳过,
                    #    改用规则#3(趋势破位 ma20<=ma60)作为「回调失败=破位」的离场信号
                    elif ALPHA_MODE != "lowvol_rev" and below_ma20 and held_days >= 2:
                        reason = f"跌破MA20({ret_pct:+.1f}%)"
                    # 5. 移动止损(M2): 自高点回撤≥TRAIL_STOP_PCT 且曾盈利≥2% 才卖
                    elif held_days > 3 and peak_ret >= 2.0 and \
                            (pos["peak"] - cur_price) / pos["peak"] >= TRAIL_STOP_PCT / 100:
                        reason = f"移动止损({ret_pct:+.1f}%)"
                    # 6. 安全上限(C1): 极长持有(>MAX_HOLD_DAYS)才强制了结, 非正常退出
                    elif held_days >= MAX_HOLD_DAYS:
                        reason = f"持仓上限({held_days}天)"

                if reason:
                    cash += pos["shares"] * cur_price * (1 - cr)  # 卖出
                    net_ret = (cur_price * (1 - cr)) / (pos["buy_price"] * (1 + cr)) - 1
                    sell_log.append({
                        "code": code, "buy_p": pos["buy_price"], "sell_p": cur_price,
                        "held": held_days, "ret": round(ret_pct, 2),
                        "net_ret": round(net_ret * 100, 2), "reason": reason
                    })
                    to_sell.append(code)

            for code in to_sell:
                del positions[code]

            # === 计算总资产 ===
            total = cash
            for pos in positions.values():
                cp = self.get_price_on_date(pos["code"], date_str) or pos["buy_price"]
                total += pos["shares"] * cp
            portfolio.append((date_str, round(total, 2)))

            # === 入场判断 ===
            if not regime.get(date_str, True):
                continue
            if bear:           # L0: 熊市不清仓外不新建仓
                continue
            if cash < 5000:  # 资金太少不买
                continue
            if len(positions) >= MAX_POSITIONS:  # M4: 持仓已满不买
                continue

            df = self.load_day_data(date_str)
            if len(df) == 0:
                continue
            df = self.filter_stocks(df)
            df = self.score_stocks(df)

            # Top N, 排除已有持仓
            top = df[~df["code"].isin(set(positions.keys()))].head(MAX_PICKS_PER_DAY * 3)
            if len(top) == 0:
                continue

            # 低波内核门槛: 当日全市场波动率中位数(仅买低于中位数的"安静票")
            vol_med = (float(df["vol20"].median())
                       if "vol20" in df.columns and df["vol20"].notna().any()
                       else float("inf"))
            # 近20日超跌门槛: 全市场 chg_20d 底部30%分位(自适应: 牛/熊市均判为超跌)
            chg20_q30 = (float(df["chg_20d"].quantile(0.30))
                         if "chg_20d" in df.columns and df["chg_20d"].notna().any()
                         else -0.05)

            buy_count = 0
            scores = top["composite_score"].values
            s_median = float(np.median(scores[scores == scores])) if len(scores) > 0 else 0
            s_max = float(np.max(scores)) if len(scores) > 0 else 1
            s_range = max(s_max - s_median, 0.01)

            for _, row in top.iterrows():
                code = row["code"]
                price = float(row["close"]) if pd.notna(row.get("close")) else 0
                if price <= 0:
                    continue

                # B1: 相对强度择时 — 只买站上 MA20 的强势股; 若 MA60 可得且趋势向下则不买
                ma20 = self.get_ma_on_date(code, date_str, 20)
                ma60 = self.get_ma_on_date(code, date_str, 60)
                if ALPHA_MODE == "lowvol_rev":
                    # --- f_trend 降为硬闸门(原正向权重已置0): 仅长趋势向上才考虑 ---
                    if ma20 is not None and ma60 is not None and ma20 <= ma60:
                        continue  # 长趋势向下(MA20≤MA60) → 不买
                    # --- f_rs 降为软闸门(不接远离MA20的飞刀, 而非"必须站上均线买强") ---
                    if ma20 is not None and price < ma20 * 0.93:
                        continue  # 已远低于MA20(自由落体), 不接飞刀
                    # --- 近20日超跌(反转核心判据): 仅买底部30%分位的超跌票 ---
                    chg20 = row.get("chg_20d")
                    if chg20 is not None and not pd.isna(chg20) and chg20 > chg20_q30:
                        continue  # 近20日未达超跌(>底部30%分位) → 非反转候选
                    # --- 自由落体兜底(MA60 长线支撑) ---
                    if ma60 is not None and price < ma60 * 0.93:
                        continue  # 跌破长线支撑, 不买
                    # --- 低波门槛(真alpha内核) ---
                    vol = row.get("vol20")
                    if vol is not None and not pd.isna(vol) and vol > vol_med:
                        continue  # 高波动票, 仅买低波动
                else:
                    if ma20 is not None and price < ma20:
                        continue  # 跌破 MA20 的弱势股, 不买
                    if ma20 is not None and ma60 is not None and ma20 <= ma60:
                        continue  # 下降趋势 (MA20 ≤ MA60), 不买
                    # 回踩不破守卫(降换手实验): 近5日曾破MA20(未站稳)则等确认后再买, 过滤假突破追高
                    if PULLBACK_GUARD and not bool(row.get("pullback_ok", True)):
                        continue

                alloc = cash * mscale / max(MAX_PICKS_PER_DAY - buy_count, 1)
                shares = int(alloc / price / 100) * 100
                if shares < 100:
                    continue

                cost = shares * price * (1 + self._trade_cost(code, is_buy=True))  # 买入
                if cost > cash * 0.5:
                    continue

                # 动态持仓天数 (M5): 5-15天
                s = float(row["composite_score"])
                percentile = min(1.0, max(0.1, (s - s_median) / s_range + 0.5))
                hold_days = max(5, min(15, int(8 + percentile * 7)))  # 5-15天

                # 目标价 (M5): 5%-12%
                target_pct = TARGET_BASE + percentile * 7  # 5%-12%
                target_price = price * (1 + target_pct / 100)
                stop_price = price * (1 - STOP_LOSS / 100)

                cash -= cost
                positions[code] = {
                    "code": code, "buy_price": price, "shares": shares,
                    "target": round(target_price, 2), "stop": round(stop_price, 2),
                    "hold_days": hold_days, "entry_idx": di, "score": round(s, 2),
                    "peak": price,
                }
                buy_count += 1
                if buy_count >= MAX_PICKS_PER_DAY:
                    break

        # 清仓 (A2: 期末平仓也计入真实买卖胜率)
        last_date = all_dates[-1]
        last_idx = len(all_dates) - 1
        for code, pos in list(positions.items()):
            sp = self.get_price_on_date(code, last_date) or pos["buy_price"]
            cr = self._trade_cost(code)
            cash += pos["shares"] * sp * (1 - cr)  # 清仓
            gross = sp / pos["buy_price"] - 1
            net_ret = (sp * (1 - cr)) / (pos["buy_price"] * (1 + cr)) - 1
            sell_log.append({
                "code": code, "buy_p": pos["buy_price"], "sell_p": sp,
                "held": last_idx - pos["entry_idx"], "ret": round(gross * 100, 2),
                "net_ret": round(net_ret * 100, 2), "reason": "期末清仓"
            })
        total = cash
        portfolio.append((last_date, round(total, 2)))

        # === 统计 ===
        values = [v for _, v in portfolio]
        returns_daily = [(values[i]/values[i-1]-1) for i in range(1,len(values)) if values[i-1]>0]
        years = len(values) / 252
        cagr = (values[-1]/cap)**(1/years)-1 if years>0 else 0

        peak = values[0]; max_dd = 0.0
        for v in values:
            if v > peak: peak = v
            dd = (peak-v)/peak if peak>0 else 0
            if dd > max_dd: max_dd = dd

        ret_arr = np.array(returns_daily) if returns_daily else np.array([0])
        rf_daily = 0.02/252
        excess = ret_arr - rf_daily
        sharpe = float(np.mean(excess)/np.std(excess)*np.sqrt(252)) if np.std(excess)>0 else 0

        # 卖出原因统计
        reasons = {}
        for sl in sell_log:
            r = sl["reason"].split("(")[0]
            reasons[r] = reasons.get(r, 0) + 1
        avg_held = np.mean([sl["held"] for sl in sell_log]) if sell_log else 0
        avg_ret = np.mean([sl["ret"] for sl in sell_log]) if sell_log else 0

        # A2: 真实买卖胜率 (扣买卖成本后的净收益)
        net_rets = [sl["net_ret"] for sl in sell_log] if sell_log else []
        wins = [r for r in net_rets if r > 0]
        losses = [r for r in net_rets if r <= 0]
        win_rate = (len(wins) / len(net_rets) * 100) if net_rets else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        pl_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")
        # 净收益分布
        dist = {"<=-5%": 0, "-5%~0%": 0, "0%~5%": 0, "5%~10%": 0, ">=10%": 0}
        for r in net_rets:
            if r <= -5: dist["<=-5%"] += 1
            elif r < 0: dist["-5%~0%"] += 1
            elif r < 5: dist["0%~5%"] += 1
            elif r < 10: dist["5%~10%"] += 1
            else: dist[">=10%"] += 1

        # 逐年
        yearly = {}
        for d, v in portfolio:
            y = d[:4]; yearly.setdefault(y, []).append(v)
        year_returns = {y: round((vals[-1]/vals[0]-1)*100,1) for y,vals in yearly.items() if vals[0]>0}

        return {
            "initial": cap, "final": round(values[-1],2),
            "return_pct": round((values[-1]/cap-1)*100,2),
            "cagr_pct": round(cagr*100,2), "max_drawdown_pct": round(max_dd*100,2),
            "sharpe": round(sharpe,2), "years": round(years,1),
            "year_returns": year_returns, "portfolio": portfolio,
            "sell_stats": {"reasons": reasons, "avg_held_days": round(avg_held,1),
                           "avg_return": round(avg_ret,2), "total_trades": len(sell_log),
                           "win_rate": round(win_rate, 1),
                           "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
                           "profit_loss_ratio": round(pl_ratio, 2) if pl_ratio != float("inf") else None,
                           "net_return_dist": dist},
            "market_gate": self._market_gate_stats(),
        }

    def _market_gate_stats(self) -> dict:
        """汇总 L0 市场闸门每日仓位系数分布(供报告展示)。"""
        if not MARKET_GATE or not hasattr(self, "_mkt_scale"):
            return {"enabled": False}
        sc = self._mkt_scale
        return {
            "enabled": True,
            "bear_days": sum(1 for v in sc.values() if v <= 0.0),
            "full_days": sum(1 for v in sc.values() if v >= 1.0),
            "light_days": sum(1 for v in sc.values() if 0 < v < 1.0),
            "total_days": len(sc),
        }


def generate_html_report(pf: dict, config: dict) -> str:
    """生成可视化 HTML 报告。"""
    portfolio = pf.get("portfolio", [])
    if not portfolio:
        return ""

    dates = [p[0][:10] for p in portfolio]
    values = [p[1] for p in portfolio]
    initial = pf["initial"]

    # 计算回撤序列
    peak = values[0]
    drawdowns = []
    for v in values:
        if v > peak: peak = v
        drawdowns.append(round((peak - v) / peak * 100, 2))

    # 逐年收益
    year_rows = "".join(
        f"<tr><td>{y}</td><td style='color:{'red' if r>0 else 'green'}'>{r:+.1f}%</td></tr>"
        for y, r in sorted(pf.get("year_returns", {}).items())
    )

    sell = pf.get("sell_stats", {})
    reason_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td>{v/sell['total_trades']*100:.0f}%</td></tr>"
        for k, v in sell.get("reasons", {}).items()
    ) if sell else ""
    dist = sell.get("net_return_dist", {})
    dist_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td>{v/sell['total_trades']*100:.0f}%</td></tr>"
        for k, v in dist.items()
    ) if dist else ""
    win_rate = sell.get("win_rate", 0)
    pl_ratio = sell.get("profit_loss_ratio")
    mg = pf.get("market_gate", {"enabled": False})

    html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"UTF-8\">
<title>A股多因子回测报告</title>
<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4\"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Microsoft YaHei',sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}}
.container{{max-width:1200px;margin:0 auto}}
h1{{text-align:center;margin:20px 0;font-size:28px;color:#38bdf8}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#1e293b;border-radius:12px;padding:20px;text-align:center;border:1px solid #334155}}
.card .label{{font-size:13px;color:#94a3b8;margin-bottom:8px}}
.card .value{{font-size:28px;font-weight:700;font-family:monospace}}
.red{{color:#ef4444}}.green{{color:#22c55e}}.blue{{color:#38bdf8}}
.chart-container{{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:20px;border:1px solid #334155}}
.chart-container h3{{margin-bottom:16px;color:#94a3b8}}
canvas{{max-height:350px}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}
th,td{{padding:10px 16px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-size:13px;text-transform:uppercase}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:768px){{.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class=\"container\">
<h1>A股多因子选股系统 — 回测报告</h1>
<p style=\"text-align:center;color:#64748b;margin-bottom:24px\">
  区间: {dates[0]} ~ {dates[-1]} | {pf['years']}年 | 
  佣金: 万0.854免5 (股票{config.get('trade',{}).get('stock_cost',0.0013)*100:.2f}% / ETF{config.get('trade',{}).get('etf_cost',0.00017)*100:.3f}%)
</p>

<div class=\"cards\">
  <div class=\"card\"><div class=\"label\">初始资金</div><div class=\"value blue\">¥{initial:,.0f}</div></div>
  <div class=\"card\"><div class=\"label\">最终资金</div><div class=\"value {'red' if pf['return_pct']>0 else 'green'}\">¥{pf['final']:,.0f}</div></div>
  <div class=\"card\"><div class=\"label\">总收益</div><div class=\"value {'red' if pf['return_pct']>0 else 'green'}\">{pf['return_pct']:+.1f}%</div></div>
  <div class=\"card\"><div class=\"label\">年化收益</div><div class=\"value {'red' if pf['cagr_pct']>0 else 'green'}\">{pf['cagr_pct']:+.1f}%</div></div>
  <div class=\"card\"><div class=\"label\">最大回撤</div><div class=\"value green\">-{pf['max_drawdown_pct']:.1f}%</div></div>
  <div class=\"card\"><div class=\"label\">夏普比率</div><div class=\"value blue\">{pf['sharpe']}</div></div>
  <div class=\"card\"><div class=\"label\">买卖胜率(净)</div><div class=\"value {'red' if win_rate<50 else 'green'}\">{win_rate:.1f}%</div></div>
  <div class=\"card\"><div class=\"label\">L0市场闸门</div><div class=\"value blue\">{('熊市'+str(mg.get('bear_days',0))+'天') if mg.get('enabled') else '关闭'}</div></div>
</div>

<div class=\"chart-container\"><h3>资产曲线</h3><canvas id=\"equityChart\"></canvas></div>

<div class=\"two-col\">
  <div class=\"chart-container\"><h3>回撤曲线</h3><canvas id=\"ddChart\"></canvas></div>
  <div class=\"chart-container\"><h3>逐年收益</h3><table>{year_rows}</table></div>
</div>

<div class=\"chart-container\"><h3>卖出原因分布</h3><table>
<tr><th>原因</th><th>笔数</th><th>占比</th></tr>{reason_rows}
<tr style=\"font-weight:700\"><td>合计</td><td>{sell['total_trades']}</td><td>平均持有{sell.get('avg_held_days',0)}天 / 均益{sell.get('avg_return',0):+.2f}%</td></tr>
</table></div>

<div class=\"two-col\">
  <div class=\"chart-container\"><h3>真实买卖胜率 (扣成本)</h3><table>
  <tr><th>指标</th><th>数值</th></tr>
  <tr><td>总交易笔数</td><td>{sell['total_trades']}</td></tr>
  <tr><td>胜率(净收益&gt;0)</td><td style='color:{'red' if win_rate<50 else 'green'};font-weight:700'>{win_rate:.1f}%</td></tr>
  <tr><td>平均盈利</td><td>{sell.get('avg_win',0):+.2f}%</td></tr>
  <tr><td>平均亏损</td><td>{sell.get('avg_loss',0):+.2f}%</td></tr>
  <tr><td>盈亏比</td><td>{pl_ratio if pl_ratio is not None else '—'}</td></tr>
  </table></div>
  <div class=\"chart-container\"><h3>净收益分布</h3><table>
  <tr><th>区间</th><th>笔数</th><th>占比</th></tr>{dist_rows}
  </table></div>
</div>

</div>
<script>
const dates = {dates};
const values = {values};
const dd = {drawdowns};
new Chart(document.getElementById('equityChart'),{{
  type:'line',data:{{labels:dates,datasets:[{{label:'资产(¥)',data:values,borderColor:'#38bdf8',borderWidth:2,pointRadius:0,fill:false,tension:0.1}}]}},
  options:{{responsive:true,scales:{{x:{{ticks:{{color:'#94a3b8',maxTicksLimit:12}}}},y:{{ticks:{{color:'#94a3b8',callback:v=>'¥'+(v/10000).toFixed(0)+'万'}}}}}},plugins:{{legend:{{display:false}}}}}}
}});
new Chart(document.getElementById('ddChart'),{{
  type:'area',data:{{labels:dates,datasets:[{{label:'回撤(%)',data:dd,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.15)',borderWidth:2,pointRadius:0,fill:true,tension:0.1}}]}},
  options:{{responsive:true,scales:{{x:{{ticks:{{color:'#94a3b8',maxTicksLimit:12}}}},y:{{reverse:true,ticks:{{color:'#94a3b8',callback:v=>v+'%'}}}}}},plugins:{{legend:{{display:false}}}}}}
}});
</script>
</body></html>"""

    tag = f"sl{STOP_LOSS:.0f}"
    if ALPHA_MODE == "lowvol_rev":
        tag += "_lvrev"
    if MIN_HOLD > 0:
        tag += f"_mh{MIN_HOLD}"
    if PULLBACK_GUARD:
        tag += "_pg"
    html_path = Path("briefs") / f"回测报告_{tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def main():
    # 加载配置
    import yaml
    config = {}
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

    def cfg(section, key, default):
        return config.get(section, {}).get(key, default)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=" * 60)
    logger.info("本地数据回测启动")
    logger.info("=" * 60)

    # ---- 命令行参数 (A/B 对比: 止损线 / 最小持有期 / 回踩守卫 / 跳过 T+N / alpha 内核) ----
    global STOP_LOSS, MIN_HOLD, PULLBACK_GUARD, ALPHA_MODE, MARKET_GATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-loss", type=float, default=STOP_LOSS,
                    help="硬止损线(%%)，覆盖默认 STOP_LOSS，用于 A/B 对比")
    ap.add_argument("--min-hold", type=int, default=MIN_HOLD,
                    help="最小持有期(天): 非硬止损卖出需持有>=N天, 覆盖默认 MIN_HOLD")
    ap.add_argument("--pullback-guard", action="store_true",
                    help="入场回踩不破守卫: 近5日最低价>=MA20 才买, 用于降换手对比")
    ap.add_argument("--alpha-mode", choices=["trend", "lowvol_rev"], default=ALPHA_MODE,
                    help="选股 alpha 内核: trend(v4.8买强/动量,对照) | lowvol_rev(低波+反转+质量, 真 alpha 路线, 默认)")
    ap.add_argument("--no-market-gate", action="store_true",
                    help="关闭 L0 市场闸门(宽基MA200+估值分位), 满仓对照, 用于 A/B 验证闸门贡献")
    ap.add_argument("--skip-tn", action="store_true",
                    help="跳过 T+N 选股验证(run)，仅跑资金模拟，省时")
    args = ap.parse_args()
    STOP_LOSS = args.stop_loss
    MIN_HOLD = max(0, args.min_hold)
    PULLBACK_GUARD = args.pullback_guard
    ALPHA_MODE = args.alpha_mode
    MARKET_GATE = not args.no_market_gate
    if ALPHA_MODE == "lowvol_rev":
        logger.info("选股内核: 低波+反转+质量 (真 alpha 路线) — 近20日超跌/低波门槛/f_rs·f_trend降为闸门")
    if not MARKET_GATE:
        logger.info("L0 市场闸门: 已关闭 (满仓对照)")
    if abs(STOP_LOSS - 8.0) > 1e-9:
        logger.info(f"覆盖止损线 STOP_LOSS = {STOP_LOSS}% (默认 8%)")
    if MIN_HOLD > 0:
        logger.info(f"启用最小持有期 MIN_HOLD = {MIN_HOLD} 天")
    if PULLBACK_GUARD:
        logger.info("启用入场回踩不破守卫 (pullback-guard)")

    bt = LocalBacktest()
    bt.config = config  # 注入配置

    try:
        # === 资金模拟 ===
        cap = cfg("portfolio", "initial_capital", INITIAL_CAPITAL)
        mpd = cfg("portfolio", "max_picks_per_day", MAX_PICKS_PER_DAY)
        logger.info("\n" + "=" * 60)
        logger.info(f"资金模拟: 初始 {cap:,} 元, 日选 {mpd} 只, 动态持有(相对强度择时)")
        logger.info("=" * 60)
        pf = bt.run_portfolio()

        print(f"\n{'='*60}")
        print(f"资金模拟结果 ({pf['years']}年)")
        print(f"{'='*60}")
        print(f"初始资金: ¥{pf['initial']:,.0f}")
        print(f"最终资金: ¥{pf['final']:,.0f}")
        print(f"总收益:   {pf['return_pct']:+.2f}%")
        print(f"年化收益: {pf['cagr_pct']:+.2f}%")
        print(f"最大回撤: {pf['max_drawdown_pct']:.1f}%")
        print(f"夏普比率: {pf['sharpe']}")

        if pf.get("year_returns"):
            print(f"\n逐年收益:")
            for y, r in sorted(pf["year_returns"].items()):
                print(f"  {y}: {r:+.1f}%")

        # 卖出统计
        sell_stats = pf.get("sell_stats", {})
        if sell_stats:
            print(f"\n卖出统计 ({sell_stats['total_trades']}笔):")
            print(f"  真实买卖胜率(净): {sell_stats.get('win_rate',0):.1f}% | "
                  f"盈亏比: {sell_stats.get('profit_loss_ratio')} | "
                  f"平均盈利 {sell_stats.get('avg_win',0):+.2f}% / 平均亏损 {sell_stats.get('avg_loss',0):+.2f}%")
            print(f"  平均持有: {sell_stats['avg_held_days']}天 | 平均收益: {sell_stats['avg_return']:+.2f}%")
            for reason, cnt in sell_stats.get("reasons", {}).items():
                print(f"  {reason}: {cnt}笔")

        # 保存资产曲线
        portfolio = pf.get("portfolio", [])
        if portfolio:
            mh_tag = f"_mh{MIN_HOLD}" if MIN_HOLD > 0 else ""
            lv_tag = "_lvrev" if ALPHA_MODE == "lowvol_rev" else ""
            csv_path = Path("briefs") / f"portfolio_curve_sl{STOP_LOSS:.0f}{lv_tag}{mh_tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(portfolio, columns=["date", "value"]).to_csv(csv_path, index=False)
            print(f"\n资产曲线: {csv_path}")

        # 生成 HTML 报告
        if cfg("output", "save_html", True):
            html_path = generate_html_report(pf, config)
            print(f"可视化: {html_path}")

        # ===== A1: 统一框架 — 同时跑 T+N 验证(run), 消除"胜率"概念混淆 =====
        # run_portfolio 产出的是「真实资金买卖」胜率(见上 sell_stats.win_rate);
        # run() 产出的是「选股后持有 T+N 日」的截面胜率, 两套口径不同, 一并呈现。
        if not args.skip_tn:
            logger.info("\n" + "=" * 60)
            logger.info("T+N 选股验证框架 (run)")
            logger.info("=" * 60)
            res = bt.run(start_date="2020-01-01")
            print(f"\n{'='*60}")
            print(f"T+N 选股验证 ({res['total_dates']}交易日, {res['total_picks']}次推荐)")
            print(f"{'='*60}")
            for period in ["T+1_ret", "T+3_ret", "T+5_ret"]:
                st = res.get("stats", {}).get(period)
                if st:
                    print(f"  {period}: 胜率 {st['win_rate']:.1f}% | 均值 {st['avg']:+.2f}% | "
                          f"样本 {st['count']} | 极值 [{st['min']:+.1f}%, {st['max']:+.1f}%]")
            rs = res.get("stats_risk", {}).get("T+1_ret")
            if rs:
                print(f"  风控后 T+1 胜率: {rs['win_rate']:.1f}% | 均值 {rs['avg']:+.2f}%")

    except KeyboardInterrupt:
        logger.warning("用户中断")
    finally:
        bt.db.close()
        logger.info("DB 已关闭")


if __name__ == "__main__":
    main()
