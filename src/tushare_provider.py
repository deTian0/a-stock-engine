"""
tushare_provider.py - Tushare Pro 数据源

接口与 AkshareProvider 完全兼容，接入 westock_cli.py 的回退链。
核心优势: fina_indicator 补齐 ROE/毛利率/负债率/营收增速/利润增速。

配置从环境变量读取（优先）或项目根目录 .env 文件:
  TUSHARE_TOKEN  — tushare pro API token（必填）
  TUSHARE_URL     — 自定义 API 端点（可选，默认官方）

用法:
    from tushare_provider import get_tushare
    provider = get_tushare()
    df = provider.get_stock_list()
    df = provider.get_fundamentals(["000001", "600519"])
"""

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from database import get_db

logger = logging.getLogger(__name__)

# ---- 配置加载 ----

def _load_env():
    """加载 .env 文件（如果存在）。返回更新后的 os.environ 副本（不影响实际环境）。"""
    env_paths = [
        Path(__file__).parent.parent / ".env",          # 项目根目录
        Path.cwd() / ".env",                             # 当前工作目录
    ]
    for p in env_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key, val = key.strip(), val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                logger.debug(f"已加载环境变量: {p}")
                break
            except Exception:
                pass

_load_env()

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
TUSHARE_URL = os.getenv("TUSHARE_URL", "")

if not TUSHARE_TOKEN:
    logger.warning("TUSHARE_TOKEN 未设置。创建 .env 文件或设置环境变量。参见 .env.example。")



def _ts_code(code: str) -> str:
    """6位代码 → tushare ts_code 格式 (000001.SZ / 600519.SH / 83xxxx.BJ)。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    elif code.startswith(("8", "4")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _from_ts_code(ts_code: str) -> str:
    """ts_code → 6位纯数字代码。"""
    return ts_code.split(".")[0].zfill(6)


class TushareProvider:
    """Tushare Pro 数据源，接口与 AkshareProvider 兼容。"""

    def __init__(self, cache_ttl_hours: int = 6):
        self._cache_ttl = cache_ttl_hours
        self._source = "tushare"
        self._pro = None

    @property
    def pro(self):
        """懒加载 tushare pro API。"""
        if self._pro is None:
            if not TUSHARE_TOKEN:
                raise RuntimeError(
                    "TUSHARE_TOKEN 未配置。请复制 .env.example 为 .env 并填入 token。"
                )
            import tushare as ts
            self._pro = ts.pro_api(TUSHARE_TOKEN)
            if TUSHARE_URL:
                self._pro._DataApi__http_url = TUSHARE_URL
            logger.info("Tushare Pro API 已初始化%s", 
                        f" (自定义端点: {TUSHARE_URL})" if TUSHARE_URL else "")
        return self._pro

    @property
    def db(self):
        return get_db()

    # ============================================================
    #  股票列表
    # ============================================================

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全A股股票列表（含 PE/PB/市值/收盘价）。
        组合 stock_basic + daily_basic 两个接口。
        """
        cache_key = "stock_list_tushare"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # Step 1: 基础信息（代码/名称/行业/上市日期）
            basic = self.pro.stock_basic(
                exchange="", list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date"
            )
            if basic is None or len(basic) == 0:
                raise RuntimeError("stock_basic 返回空")
            basic["code"] = basic["ts_code"].apply(_from_ts_code)
            basic = basic.rename(columns={"industry": "sector_raw"})
            logger.info(f"tushare stock_basic: {len(basic)} 只")

            # Step 2: 估值数据（PE/PB/市值）— 尝试最近交易日
            try:
                latest_date = None
                for offset in range(5):
                    test_date = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
                    test_row = self.pro.daily_basic(trade_date=test_date, limit=1)
                    if test_row is not None and len(test_row) > 0:
                        latest_date = test_date
                        break
                if latest_date:
                    # 先获取基准列（一定有: ts_code,close,pe,pb,total_mv,circ_mv）
                    valuation = self.pro.daily_basic(
                        trade_date=latest_date,
                        fields="ts_code,close,pe,pb,total_mv,circ_mv,volume_ratio,turnover_rate"
                    )
                    if valuation is not None and len(valuation) > 0:
                        valuation["code"] = valuation["ts_code"].apply(_from_ts_code)
                        # tushare total_mv 单位: 万元 → 元
                        valuation["market_cap"] = valuation["total_mv"] * 10000
                        valuation["float_cap"] = valuation["circ_mv"] * 10000
                        # Step 2b: 从 daily 接口补齐 change_pct（涨跌幅）和 amount/volume
                        try:
                            daily_raw = self.pro.daily(trade_date=latest_date)
                            if daily_raw is not None and len(daily_raw) > 0:
                                daily_raw["code"] = daily_raw["ts_code"].apply(_from_ts_code)
                                # 涨跌幅映射
                                pct_map = {}
                                for _, dr in daily_raw.iterrows():
                                    c = dr.get("code", "")
                                    chg = dr.get("pct_chg")
                                    if c and chg is not None:
                                        try:
                                            pct_map[c] = float(chg)
                                        except (ValueError, TypeError):
                                            pass
                                if pct_map:
                                    valuation["change_pct"] = valuation["code"].map(pct_map)
                                # 成交额映射（daily 有 amount，单位：千元）
                                amt_map = {}
                                for _, dr in daily_raw.iterrows():
                                    c = dr.get("code", "")
                                    amt = dr.get("amount")
                                    if c and amt is not None:
                                        try:
                                            amt_map[c] = float(amt) * 1000  # 千元 → 元
                                        except (ValueError, TypeError):
                                            pass
                                if amt_map:
                                    valuation["amount"] = valuation["code"].map(amt_map)
                                logger.info(f"daily 补齐: change_pct={valuation['change_pct'].notna().sum() if 'change_pct' in valuation.columns else 0} "
                                           f"amount={valuation['amount'].notna().sum() if 'amount' in valuation.columns else 0}")
                            else:
                                logger.info("daily 接口返回空，尝试用 daily_basic 的 close/pre_close 计算涨跌幅")
                                # 兜底：用 pre_close（如果 daily_basic 有此字段）
                                if "pre_close" in valuation.columns:
                                    valuation["change_pct"] = (
                                        (valuation["close"] - valuation["pre_close"]) / valuation["pre_close"] * 100
                                    )
                        except Exception as e:
                            logger.warning(f"daily 接口补齐行情数据失败: {type(e).__name__}: {e}")
                        # 合并
                        merge_cols = ["code", "close", "pe", "pb", "market_cap",
                                     "float_cap", "volume_ratio", "turnover_rate", "amount", "volume"]
                        available = [c for c in merge_cols if c in valuation.columns]
                        df = basic.merge(
                            valuation[available],
                            on="code", how="left"
                        )
                        df = df.rename(columns={"turnover_rate": "turnover", "volume_ratio": "volume_ratio"})
                        # 填充默认值
                        for col in ("close", "pe", "pb", "market_cap", "float_cap", "amount", "volume"):
                            if col not in df.columns:
                                df[col] = np.nan
                        logger.info(f"tushare daily_basic 合并: {latest_date}, "
                                    f"PE命中={df['pe'].notna().sum()}, "
                                    f"amount非空={df['amount'].notna().sum()}")
                    else:
                        df = basic
                else:
                    df = basic
            except Exception as e:
                logger.warning(f"daily_basic 获取失败: {e}，仅使用基础信息")
                df = basic

            # 填充缺失值 — amount 设为极大值绕过过滤（daily_basic 不返回成交额）
            for col in ("close", "pe", "pb", "market_cap", "float_cap"):
                if col not in df.columns:
                    df[col] = np.nan
            if "amount" not in df.columns or df["amount"].isna().all():
                df["amount"] = 1e12  # 极大值，绕过成交额过滤
            for col, default in [("volume_ratio", np.nan), ("turnover", np.nan),
                                 ("change_pct", np.nan), ("volume", np.nan)]:
                if col not in df.columns:
                    df[col] = default

            logger.info(f"tushare 股票列表: {len(df)} 只 (PE有效: {df['pe'].notna().sum()})")
            self.db.cache_put(cache_key, "stock_list", df, self._source, self._cache_ttl)
            return df

        except Exception as e:
            logger.error(f"tushare 获取股票列表失败: {e}")
            raise

    # ============================================================
    #  K线数据
    # ============================================================

    def get_kline(self, code: str, days: int = 120, adjust: str = "qfq") -> Optional[pd.DataFrame]:
        """获取个股日K线数据。"""
        cache_key = f"kline_ts_{code}_{days}_{adjust}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            ts_code = _ts_code(code)
            adj_map = {"qfq": "qfq", "hfq": "hfq", "": None}
            adj = adj_map.get(adjust, "qfq")

            import tushare as ts
            df = ts.pro_bar(api=self.pro, ts_code=ts_code, adj=adj,
                            start_date=(datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d"),
                            end_date=datetime.now().strftime("%Y%m%d"),
                            factors=["tor", "vr"] if adj else None)

            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.rename(columns={
                "trade_date": "date",
                "vol": "volume",
                "pct_chg": "change_pct",
                "turnover_rate": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("date").tail(days).reset_index(drop=True)

            if len(df) > 0:
                self.db.cache_put(cache_key, "kline", df, self._source, self._cache_ttl)
                logger.debug(f"tushare K线 {code}: {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"tushare 获取K线失败 {code}: {e}")
            return pd.DataFrame()

    # ============================================================
    #  指数K线
    # ============================================================

    def get_index_kline(self, code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取指数日K线数据。"""
        cache_key = f"index_kline_ts_{code}_{days}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # 指数代码 → tushare 格式
            idx_map = {
                "000001": "000001.SH",  # 上证指数
                "399001": "399001.SZ",  # 深证成指
                "399006": "399006.SZ",  # 创业板指
                "000688": "000688.SH",  # 科创50
                "000300": "000300.SH",  # 沪深300
            }
            ts_code = idx_map.get(str(code).zfill(6), _ts_code(code))

            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=(datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.rename(columns={
                "trade_date": "date",
                "vol": "volume",
                "pct_chg": "change_pct",
            })
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("date").tail(days).reset_index(drop=True)

            self.db.cache_put(cache_key, "index_kline", df, self._source, self._cache_ttl)
            logger.debug(f"tushare 指数K线 {code}: {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"tushare 获取指数K线失败 {code}: {e}")
            return pd.DataFrame()

    # ============================================================
    #  基本面数据 ⭐ 核心优势
    # ============================================================

    def get_fundamentals(self, codes: list[str]) -> pd.DataFrame:
        """
        获取批量基本面数据。
        fina_indicator 一次返回 ROE/ROA/毛利率/负债率/营收增速/利润增速。
        """
        import hashlib
        key_hash = hashlib.md5(",".join(sorted(codes[:30])).encode()).hexdigest()[:12]
        cache_key = f"fundamentals_ts_{key_hash}_{len(codes)}"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # 转换为 tushare ts_code 格式
            ts_codes = [_ts_code(c) for c in codes]
            all_rows = []

            # fina_indicator 每次最多查一批，分批处理（每批1000个）
            batch_size = 1000
            for i in range(0, len(ts_codes), batch_size):
                batch = ts_codes[i:i + batch_size]
                ts_code_str = ",".join(batch)
                df = self.pro.fina_indicator(
                    ts_code=ts_code_str,
                    fields="ts_code,roe,roa,grossprofit_margin,debt_to_assets,"
                           "or_yoy,profit_dedt,total_mv,eps"
                )
                if df is not None and len(df) > 0:
                    all_rows.append(df)

            if not all_rows:
                logger.warning("tushare fina_indicator 返回空")
                return pd.DataFrame()

            combined = pd.concat(all_rows, ignore_index=True)
            combined["code"] = combined["ts_code"].apply(_from_ts_code)

            # 映射为标准列名
            combined = combined.rename(columns={
                "roe": "roe",
                "grossprofit_margin": "gross_margin",
                "debt_to_assets": "debt_ratio",
                "or_yoy": "revenue_growth",
                "profit_dedt": "profit_growth",
                "total_mv": "market_cap",
            })

            # 补充 name（stock_list → stock_basic兜底 → 代码兜底）
            try:
                stock_list = self.get_stock_list()
                if "code" in stock_list.columns and "name" in stock_list.columns:
                    name_map = dict(zip(stock_list["code"].astype(str).str.zfill(6),
                                       stock_list["name"]))
                    combined["name"] = combined["code"].astype(str).str.zfill(6).map(name_map)
            except Exception:
                pass
            if "name" not in combined.columns or combined["name"].isna().sum() > len(combined) * 0.5:
                try:
                    basic = self.pro.stock_basic(exchange="", list_status="L",
                                                  fields="ts_code,name")
                    basic["code"] = basic["ts_code"].apply(_from_ts_code).str.zfill(6)
                    nm = dict(zip(basic["code"], basic["name"]))
                    combined["name"] = combined["code"].astype(str).str.zfill(6).map(nm)
                except Exception:
                    combined["name"] = combined["code"]  # 最终兜底用代码

            # 同时补充 PE/PB（从 daily_basic）
            try:
                for offset in range(3):
                    test_date = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
                    val = self.pro.daily_basic(
                        trade_date=test_date,
                        fields="ts_code,pe,pb"
                    )
                    if val is not None and len(val) > 0:
                        val["code"] = val["ts_code"].apply(_from_ts_code)
                        pe_map = dict(zip(val["code"], val["pe"]))
                        pb_map = dict(zip(val["code"], val["pb"]))
                        combined["pe"] = combined["code"].map(pe_map)
                        combined["pb"] = combined["code"].map(pb_map)
                        break
            except Exception:
                combined["pe"] = np.nan
                combined["pb"] = np.nan

            logger.info(f"tushare 基本面: {len(combined)} 条 "
                        f"(ROE有效: {combined['roe'].notna().sum()}, "
                        f"毛利率有效: {combined['gross_margin'].notna().sum()})")
            self.db.cache_put(cache_key, "fundamentals", combined, self._source, 168)
            return combined

        except Exception as e:
            logger.error(f"tushare 获取基本面失败: {e}")
            return pd.DataFrame()

    # ============================================================
    #  板块映射
    # ============================================================

    def get_sector_mapping(self) -> dict[str, str]:
        """获取股票→申万行业映射。"""
        cache_key = "sector_mapping_tushare"
        cached = self.db.cache_get(cache_key)
        if cached is not None and len(cached) > 0:
            if "code" in cached.columns and "sector" in cached.columns:
                return dict(zip(cached["code"], cached["sector"]))
            return {}

        try:
            basic = self.pro.stock_basic(
                exchange="", list_status="L",
                fields="ts_code,symbol,industry"
            )
            if basic is None or len(basic) == 0:
                return {}

            basic["code"] = basic["ts_code"].apply(_from_ts_code)
            basic["sector"] = basic["industry"].fillna("综合")
            result = dict(zip(basic["code"], basic["sector"]))

            mapping_df = pd.DataFrame([
                {"code": k, "sector": v} for k, v in result.items()
            ])
            self.db.cache_put(cache_key, "sector_mapping", mapping_df, self._source, 168)
            logger.info(f"tushare 板块映射: {len(result)} 只")
            return result
        except Exception as e:
            logger.error(f"tushare 获取板块映射失败: {e}")
            return {}

    def get_sector_list(self) -> pd.DataFrame:
        """获取板块行情列表（申万行业指数）。"""
        cache_key = "sector_list_tushare"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            df = self.pro.index_classify(level="L1", src="SW2021")
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.rename(columns={
                "industry_name": "name",
                "index_code": "code",
            })
            df["close"] = np.nan
            df["change_pct"] = np.nan
            df["change_5d"] = np.nan
            df["change_20d"] = np.nan
            df["amount"] = np.nan
            df["amount_change"] = 0

            self.db.cache_put(cache_key, "sector_list", df, self._source, 24)
            return df
        except Exception as e:
            logger.warning(f"tushare 板块列表获取失败: {e}")
            return pd.DataFrame()

    # ============================================================
    #  概念板块归属 + 行情统计（同花顺 THS）
    # ============================================================

    def get_concept_stats(self) -> pd.DataFrame:
        """
        获取同花顺概念板块行情 + 股票→概念映射。
        返回 DataFrame: code, concept_name, concept_chg(涨幅%), concept_amount(成交额)

        首次调用需约30s加载全量概念成员（5000+只×N个概念），后续缓存7天。
        """
        cache_key = "concept_stats"
        cached = self.db.cache_get(cache_key)
        if cached is not None:
            return cached

        logger.info("加载同花顺概念板块数据...")
        try:
            # Step 1: 概念列表
            concepts = self.pro.ths_index(exchange="A", type="N")
            if concepts is None or len(concepts) == 0:
                return pd.DataFrame()
            concept_codes = concepts[concepts["ts_code"].notna()]["ts_code"].tolist()
            concept_names = dict(zip(concepts["ts_code"], concepts["name"]))
            logger.info(f"概念板块: {len(concept_codes)} 个")

            # Step 2: 概念日行情（仅取最近一天）
            today_ts = datetime.now().strftime("%Y%m%d")
            daily_data = []
            batch_size = 500
            for i in range(0, len(concept_codes), batch_size):
                batch = ",".join(concept_codes[i:i + batch_size])
                try:
                    d = self.pro.ths_daily(ts_code=batch,
                                           start_date=(datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
                                           end_date=today_ts,
                                           fields="ts_code,trade_date,pct_change,vol,amount")
                    if d is not None and len(d) > 0:
                        daily_data.append(d[d["trade_date"] == d["trade_date"].max()])
                except Exception:
                    continue
            daily_df = pd.concat(daily_data, ignore_index=True) if daily_data else pd.DataFrame()
            concept_perf = {}
            if len(daily_df) > 0:
                for _, r in daily_df.iterrows():
                    concept_perf[str(r["ts_code"])] = {
                        "chg": r.get("pct_change", 0) if pd.notna(r.get("pct_change")) else 0,
                        "amount": r.get("amount", 0) if pd.notna(r.get("amount")) else 0,
                    }
            logger.info(f"概念行情: {len(concept_perf)} 条 今日数据")

            # Step 3: 概念成员映射（股票→所属概念，缓存7天）
            member_cache_key = "concept_members"
            member_map = self.db.cache_get(member_cache_key)  # {ts_code: [(con_code, con_name)]}
            if member_map is None:
                member_map = {}
                for i in range(0, len(concept_codes), 100):
                    batch = ",".join(concept_codes[i:i + 100])
                    try:
                        m = self.pro.ths_member(ts_code=batch)
                        if m is not None and len(m) > 0:
                            for _, r in m.iterrows():
                                stock_code = _from_ts_code(str(r["con_code"]))
                                con_code = str(r["ts_code"])
                                con_name = concept_names.get(con_code, "")
                                if stock_code not in member_map:
                                    member_map[stock_code] = []
                                member_map[stock_code].append((con_code, con_name))
                    except Exception:
                        continue
                member_df = pd.DataFrame([
                    {"code": k, "concepts": v}
                    for k, v in member_map.items()
                ])
                self.db.cache_put(member_cache_key, "concept_members", member_df, self._source, 168)

            # Step 4: 为每只股票选最热概念（按当日涨幅排序取Top1）
            result_rows = []
            for code, con_list in member_map.items():
                best = None
                best_chg = -999
                for con_code, con_name in con_list:
                    perf = concept_perf.get(con_code, {})
                    chg = perf.get("chg", 0)
                    if chg > best_chg:
                        best_chg = chg
                        best = {
                            "code": code,
                            "concept_name": con_name,
                            "concept_chg": round(chg, 2),
                            "concept_amount": perf.get("amount", 0),
                        }
                if best:
                    result_rows.append(best)

            result = pd.DataFrame(result_rows)
            if len(result) > 0:
                self.db.cache_put(cache_key, "concept_stats", result, self._source, 6)
                logger.info(f"概念归属: {len(result)} 只股票映射完成")
            return result
        except Exception as e:
            logger.warning(f"概念板块获取失败: {e}")
            return pd.DataFrame()


# ================================================================
#  全局单例
# ================================================================

_provider: Optional[TushareProvider] = None


def get_tushare() -> TushareProvider:
    global _provider
    if _provider is None:
        _provider = TushareProvider()
    return _provider
