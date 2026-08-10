"""
multifactor.py - A股多因子选股引擎（核心）

分层过滤架构:
  L0 → 市场环境判断（多头/震荡/空头），决定仓位上限
  L2 → 基础过滤（ST/停牌/新股/市值/PE/成交额）
  L4 → 多因子评分排序（价值+质量+成长+动量）

输出分类:
  ②A 质量榜 - 综合评分最高的N只
  ②B 短线榜 - 短期动量最强的N只
  ③A 持仓   - 当前账户持仓
  ③B 操作   - 需要卖出/减仓的持仓
  ③C 观察名单 - 评分靠前的候选

v5 反弹引擎:
  识别超跌+放量反弹的个股，作为短线补充
"""

import sys
import os
import logging
import yaml
import time
import signal
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from westock_cli import get_cli, sector_of
from local_price_loader import LocalPriceLoader
from database import get_db, get_market_db
from enrich_short import enrich

logger = logging.getLogger(__name__)

# 线程池关闭标志（信号安全）
_executor_shutdown = threading.Event()


def _signal_handler(signum, frame):
    """信号处理器：设置关闭标志，触发线程池优雅退出。"""
    logger.warning(f"收到信号 {signum}，正在关闭线程池...")
    _executor_shutdown.set()


# 在非主线程环境中注册可能失败（如 IDE 调试模式），忽略即可
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except (ValueError, OSError):
    pass


class MultiFactorEngine:
    """多因子选股引擎。"""

    def __init__(self, config_path: str = "config/config.yaml", config_dict: dict = None):
        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._load_config(config_path)
        self.cli = get_cli()
        self.price_loader = LocalPriceLoader(
            cache_dir=self.config["data_source"]["cache_dir"]
        )
        self._sector_mapping: Optional[dict] = None

    def _load_config(self, path: str) -> dict:
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def sector_mapping(self) -> dict:
        """懒加载板块映射。"""
        if self._sector_mapping is None:
            try:
                self._sector_mapping = self.cli.get_sector_mapping()
            except Exception as e:
                logger.warning(f"加载板块映射失败: {e}")
                self._sector_mapping = {}
        return self._sector_mapping

    # ========================================
    #  L0: 市场环境判断
    # ========================================

    def assess_regime(self) -> dict:
        """
        判断当前市场环境（多头/震荡/空头）。
        基于主要指数的均线位置关系。

        Returns:
            {
                "regime": "多头" / "震荡" / "空头",
                "position_cap": float,  # 仓位上限
                "indices": {code: {name, close, ma_short, ma_long, above_ma}},
                "judgment": str,  # 判定说明
            }
        """
        cfg = self.config["env_regime"]
        indices_cfg = cfg["indices"]
        ma_short = cfg["ma_short"]
        ma_long = cfg["ma_long"]

        above_count = 0
        total = len(indices_cfg)
        index_details = {}

        for idx in indices_cfg:
            code = idx["code"]
            name = idx["name"]
            try:
                df = self.cli.get_index_kline(code, days=ma_long + 10)
                if df is None or len(df) < ma_long:
                    logger.warning(f"指数数据不足: {name} ({code}), 获取到 {len(df) if df is not None else 0} 条")
                    index_details[code] = {
                        "name": name,
                        "close": None,
                        "ma_short": None,
                        "ma_long": None,
                        "above_ma": None,
                        "error": "数据不足"
                    }
                    continue

                close = float(df["close"].iloc[-1])
                ma_s = float(df["close"].tail(ma_short).mean())
                ma_l = float(df["close"].tail(ma_long).mean())
                is_above = close > ma_l

                if is_above:
                    above_count += 1

                index_details[code] = {
                    "name": name,
                    "close": round(close, 2),
                    "ma_short": round(ma_s, 2),
                    "ma_long": round(ma_l, 2),
                    "above_ma": is_above,
                }
                logger.info(f"  {name}: 收盘={close:.2f}, MA{ma_short}={ma_s:.2f}, MA{ma_long}={ma_l:.2f}, 站上均线={'是' if is_above else '否'}")

            except Exception as e:
                logger.error(f"获取指数数据失败 {name} ({code}): {e}")
                index_details[code] = {"name": name, "error": str(e)}

        # 判定市场环境
        valid_count = sum(1 for v in index_details.values() if "error" not in v)
        if valid_count == 0:
            logger.warning("所有指数数据获取失败，默认震荡环境")
            regime = "震荡"
            judgment = "指数数据全部获取失败，默认判定为震荡"
        else:
            above_ratio = above_count / valid_count if valid_count > 0 else 0
            if above_ratio >= cfg["regime_thresholds"]["bull_above_ma_ratio"]:
                regime = "多头"
                judgment = f"{above_count}/{valid_count} 个指数站上MA{ma_long}，判定为多头"
            elif above_ratio <= (1 - cfg["regime_thresholds"]["bear_below_ma_ratio"]):
                regime = "空头"
                judgment = f"{above_count}/{valid_count} 个指数站上MA{ma_long}，判定为空头"
            else:
                regime = "震荡"
                judgment = f"{above_count}/{valid_count} 个指数站上MA{ma_long}，判定为震荡"

        position_cap = cfg["position_caps"].get(regime, 0.50)

        logger.info(f"L0 环境判断: {regime} (仓位上限 {position_cap:.0%}) — {judgment}")

        return {
            "regime": regime,
            "position_cap": position_cap,
            "indices": index_details,
            "judgment": judgment,
        }

    # ========================================
    #  L2: 基础过滤
    # ========================================

    def filter_l2(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """
        L2 基础过滤：排除 ST、停牌、新股、小市值、异常PE、低成交额。
        """
        cfg = self.config["filter_l2"]
        original_count = len(stock_list)
        df = stock_list.copy()

        # 排除 ST
        if cfg.get("exclude_st", True) and "name" in df.columns:
            df = df[~df["name"].str.contains(r"ST|\*ST", na=False, regex=True)]

        # 排除停牌
        if cfg.get("exclude_suspended", True):
            for col in ["status", "trade_status"]:
                if col in df.columns:
                    df = df[df[col].astype(str).str.contains("正常|交易|1", na=False)]

        # 排除新股（上市不足N天）
        if cfg.get("exclude_new", True) and "list_date" in df.columns:
            min_date = datetime.now() - timedelta(days=60)
            try:
                df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
                df = df[df["list_date"].isna() | (df["list_date"] < min_date)]
            except Exception as e:
                logger.warning(f"上市日期过滤失败: {e}")

        # 市值过滤
        min_cap = cfg.get("min_market_cap", 0)
        if min_cap > 0 and "market_cap" in df.columns:
            df = df[df["market_cap"] >= min_cap]

        # PE 过滤
        max_pe = cfg.get("max_pe", 9999)
        if max_pe < 9999 and "pe" in df.columns:
            df = df[(df["pe"] > 0) & (df["pe"] <= max_pe)]

        # 成交额过滤
        min_amount = cfg.get("min_daily_amount", 0)
        if min_amount > 0 and "amount" in df.columns:
            df = df[df["amount"] >= min_amount * 1e8]  # 转为元

        filtered_count = len(df)
        logger.info(f"L2 过滤: {original_count} → {filtered_count} (排除 {original_count - filtered_count})")
        return df.reset_index(drop=True)

    # ========================================
    def _extract_fundamentals(self, fund_row) -> dict:
        """从单行基本面数据提取因子值。缺失/异常值返回 NaN（排名时排末尾）。"""
        row = fund_row
        
        def _safe_float(key, default=np.nan, min_val=None, max_val=None):
            """安全提取浮点数，无效时返回 NaN。"""
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            try:
                v = float(val)
                if min_val is not None and v < min_val:
                    return default
                if max_val is not None and v > max_val:
                    return max_val
                return v
            except (ValueError, TypeError):
                return default

        return {
            "pe": _safe_float("pe", min_val=0.01),        # PE=0 无效，排除
            "pb": _safe_float("pb", min_val=0.01),
            "roe": _safe_float("roe"),
            "gross_margin": _safe_float("gross_margin"),
            "debt_ratio": _safe_float("debt_ratio", min_val=0, max_val=100),
            "revenue_growth": _safe_float("revenue_growth"),
            "profit_growth": _safe_float("profit_growth"),
            "market_cap": _safe_float("market_cap", min_val=0),
            "name": str(row.get("name", "")) if "name" in row else "",
        }

    def _load_local_snapshot(self) -> Optional[pd.DataFrame]:
        """
        从 market.db 加载最新一期本地快照数据（后复权截面）。
        忽略过期时间（历史截面数据永不过期）。
        自动标准化 code 列为 6 位字符串。
        """
        try:
            mdb = get_market_db()
            row = mdb.conn.execute("""
                SELECT cache_key, data_json FROM market_data_cache
                WHERE data_type='daily_snapshot'
                ORDER BY cache_key DESC LIMIT 1
            """).fetchone()
            if row is None:
                logger.info("本地快照: market.db 中无 daily_snapshot 数据")
                return None
            cache_key = row["cache_key"]
            data_json = row["data_json"]
            if not data_json:
                logger.info(f"本地快照 {cache_key} JSON 为空")
                return None
            from io import StringIO
            df = pd.read_json(StringIO(data_json), orient="records")
            if df is None or len(df) == 0:
                logger.info(f"本地快照 {cache_key} 解析为空")
                return None
            
            # 标准化 code 列: 确保是 6 位字符串（如 "000001"）
            if "code" in df.columns:
                df["code"] = df["code"].astype(str).str.replace(r"\.(SZ|SH|BJ)$", "", regex=True).str.zfill(6)
            
            logger.info(f"本地快照命中: {cache_key}, {len(df)} 行 (code 示例: {df['code'].iloc[0] if len(df) > 0 else 'N/A'})")
            return df
        except Exception as e:
            logger.warning(f"加载本地快照失败 [{type(e).__name__}]: {e}")
            return None

    def _batch_calc_momentum(self, codes: list[str]) -> dict[str, dict]:
        """
        从 market.db 的 daily_price 表批量计算 20日/60日动量。
        daily_price 中 code 格式为 "000001.SZ"，自动去后缀匹配。
        """
        if not codes:
            return {}
        try:
            mdb = get_market_db()
            from collections import defaultdict
            prices = defaultdict(list)
            batch_size = 500

            for i in range(0, len(codes), batch_size):
                batch = codes[i:i + batch_size]
                # 同时匹配纯数字和带后缀的格式
                batch_plain = [c for c in batch]
                batch_with_sz = [f"{c}.SZ" for c in batch]
                batch_with_sh = [f"{c}.SH" for c in batch]
                all_patterns = batch_plain + batch_with_sz + batch_with_sh
                placeholders = ",".join("?" for _ in all_patterns)
                rows = mdb.conn.execute(f"""
                    SELECT code, date, close FROM daily_price
                    WHERE code IN ({placeholders})
                    ORDER BY code, date DESC
                """, all_patterns).fetchall()
                for r in rows:
                    # 去掉交易所后缀统一匹配
                    code_clean = r["code"].split(".")[0].zfill(6)
                    prices[code_clean].append(r["close"])

            if not prices:
                logger.info("daily_price 表无数据，无法批量计算动量")
                return {}

            results = {}
            hit_count = 0
            for code in codes:
                closes = prices.get(code, [])
                if len(closes) >= 61:
                    results[code] = {
                        "momentum_20d": round((closes[0] / closes[20] - 1) * 100, 2),
                        "momentum_60d": round((closes[0] / closes[60] - 1) * 100, 2),
                    }
                    hit_count += 1
                elif len(closes) >= 21:
                    results[code] = {
                        "momentum_20d": round((closes[0] / closes[20] - 1) * 100, 2),
                        "momentum_60d": 0.0,
                    }
                    hit_count += 1
                else:
                    results[code] = {"momentum_20d": 0.0, "momentum_60d": 0.0}
            logger.info(f"批量动量计算: {hit_count}/{len(codes)} 只有效 (daily_price 命中)")
            return results
        except Exception as e:
            logger.warning(f"批量动量计算失败: {e}")
            return {}

    def _fetch_stock_data(self, code: str, fund_lookup: dict, sector_map: dict,
                          batch_momentum: dict = None) -> dict:
        """获取单只股票的数据（基本面+动量+板块），用于并发调用。
        
        动量数据优先从 batch_momentum（DB 批量预取）获取，
        失败时回退到逐只 CLI 调用。
        """
        factor_values = {"code": code}

        # 从预构建的基本面字典中提取
        if code in fund_lookup:
            factor_values.update(fund_lookup[code])

        # 动量数据: 优先批量 DB 预取，回退 CLI
        if batch_momentum and code in batch_momentum:
            factor_values.update(batch_momentum[code])
        else:
            # 回退: 逐只 CLI 获取 K 线计算动量
            try:
                price_df = self.price_loader.get_price(code, days=70)
                if len(price_df) >= 60:
                    factor_values["momentum_20d"] = self.price_loader.calc_momentum(price_df, 20)
                    factor_values["momentum_60d"] = self.price_loader.calc_momentum(price_df, 60)
                else:
                    factor_values["momentum_20d"] = 0.0
                    factor_values["momentum_60d"] = 0.0
            except Exception:
                factor_values["momentum_20d"] = 0.0
                factor_values["momentum_60d"] = 0.0

        # 板块信息 — 优先映射，回退代码前缀推断
        sector = sector_map.get(code, "")
        if not sector or sector in ("未知", "", None) or (isinstance(sector, float) and pd.isna(sector)):
            prefix = str(code)[:2] if len(str(code)) >= 2 else ""
            sector = {"60":"沪市", "68":"科创板", "00":"深市", "30":"创业板", "00":"深市主板",
                      "83":"北交所", "43":"北交所", "87":"北交所", "92":"北交所"}.get(prefix, "未知")
        factor_values["sector"] = sector

        return factor_values

    #  L4: 多因子评分
    # ========================================

    def score_l4(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        L4 多因子评分：对通过 L2 的股票进行综合评分排序。

        评分流程:
        1. 优先从 market.db 加载本地快照（基本面 + 多周期动量）
        2. 批量从 daily_price 计算 20d/60d 动量
        3. CLI 基本面 / 逐只 K 线作为回退
        4. 计算各因子值 + 加权综合评分
        """
        cfg = self.config["factor_l4"]
        factors_cfg = cfg["factors"]
        top_n = cfg.get("top_n", 30)

        if len(candidates) == 0:
            logger.warning("L4 无候选股票")
            return pd.DataFrame()

        codes = [str(c).zfill(6) for c in (candidates["code"].tolist() if "code" in candidates.columns else [])]
        total = len(codes)
        logger.info(f"L4 评分: 开始处理 {total} 只股票...")

        # === Step 1: 尝试本地快照（基本面 + 多周期动量） ===
        fund_lookup = {}
        snapshot_df = self._load_local_snapshot()
        if snapshot_df is not None and len(snapshot_df) > 0:
            codes_set = set(codes)
            hit = 0
            for _, row in snapshot_df.iterrows():
                code = str(row.get("code", ""))
                if code in codes_set:
                    fund_lookup[code] = self._extract_fundamentals(row)
                    hit += 1
            logger.info(f"  本地快照基本面: {hit}/{total} 只命中")
        else:
            logger.info("  本地快照不可用，将使用 CLI 获取基本面")

        # === Step 2: 批量计算动量（从 daily_price 表） ===
        batch_momentum = self._batch_calc_momentum(codes)

        # === Step 3: CLI 补充（仅对快照中缺失的股票） ===
        missing_codes = [c for c in codes if c not in fund_lookup]
        if missing_codes:
            try:
                fundamentals = self.cli.get_fundamentals(missing_codes)
                logger.info(f"  CLI 基本面补充: {len(fundamentals)} 条 (缺失 {len(missing_codes)} 只)")
                if len(fundamentals) > 0 and "code" in fundamentals.columns:
                    for _, row in fundamentals.iterrows():
                        code = str(row.get("code", ""))
                        fund_lookup[code] = self._extract_fundamentals(row)
            except Exception as e:
                logger.warning(f"CLI 基本面获取失败: {e}")

        # === Step 4: 预加载板块映射 ===
        sector_map = self.sector_mapping

        # === Step 5: 从候选人补齐缺失名称（tushare fina_indicator 不含名称） ===
        if "code" in candidates.columns and "name" in candidates.columns:
            name_map = {}
            for _, r in candidates.iterrows():
                n = r.get("name", "")
                if n and str(n).lower() not in ("nan", "none", ""):
                    name_map[str(r["code"]).zfill(6)] = n
            fixed = 0
            for code, info in fund_lookup.items():
                n = info.get("name", "")
                if not n or str(n).lower() in ("nan", "none", ""):
                    info["name"] = name_map.get(str(code).zfill(6), code)
                    fixed += 1
            if fixed > 0:
                logger.debug(f"  L4 名称补齐: {fixed} 只")

        # === Step 6: 并发获取数据（无 CLI 动量调用时更快） ===
        max_workers = min(self.config.get("concurrency", {}).get("max_workers", 4), total)
        logger.info(f"  并发组装因子: {max_workers} workers, {total} 只股票...")

        factor_list = []
        completed = 0
        futures_map = {}

        executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            _executor_shutdown.clear()

            for code in codes:
                future = executor.submit(
                    self._fetch_stock_data, code, fund_lookup, sector_map, batch_momentum
                )
                futures_map[future] = code
                if len(futures_map) <= max_workers:
                    delay = self.config.get("concurrency", {}).get("fetch_delay_ms", 50) / 1000.0
                    if delay > 0:
                        time.sleep(delay)

            for future in as_completed(futures_map):
                if _executor_shutdown.is_set():
                    logger.warning("收到中断信号，取消剩余任务...")
                    for f in futures_map:
                        f.cancel()
                    executor.shutdown(wait=False)
                    break

                try:
                    result = future.result(timeout=0.1)
                    factor_list.append(result)
                except Exception as e:
                    code = futures_map[future]
                    logger.debug(f"并发取数失败 {code}: {e}")

                completed += 1
                if completed % 500 == 0 or completed == total:
                    logger.info(f"  L4 进度: {completed}/{total}")

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt: 取消所有剩余任务...")
            for f in futures_map:
                f.cancel()
            executor.shutdown(wait=False)
            raise

        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                for f in list(futures_map.keys()):
                    f.cancel()
                executor.shutdown(wait=False)

        # === Step 6: 计算综合评分 ===
        scores_df = pd.DataFrame(factor_list)
        scores_df = self._calc_composite_score(scores_df, factors_cfg)

        # 按评分降序排列，取前N
        scores_df = scores_df.sort_values("composite_score", ascending=False).head(top_n)

        logger.info(f"L4 评分完成: {len(scores_df)} 只入选 (top {top_n})")
        return scores_df.reset_index(drop=True)

    def _calc_composite_score(self, df: pd.DataFrame, factors_cfg: list) -> pd.DataFrame:
        """计算加权综合评分。自动跳过缺失列并报告可用性。"""
        df = df.copy()
        df["composite_score"] = 0.0

        # 诊断: 列出实际可用的因子列
        available_cols = [c for c in df.columns if c not in ("code", "sector", "composite_score")]
        configured_names = [f["name"] for f in factors_cfg]
        missing = [n for n in configured_names if n not in df.columns]
        matched = [n for n in configured_names if n in df.columns]
        logger.info(f"  L4 因子诊断: 配置={len(configured_names)}个, 可用={len(available_cols)}列, "
                    f"命中={len(matched)}个, 缺失={missing}")

        total_weight = 0.0
        for factor in factors_cfg:
            name = factor["name"]
            weight = factor["weight"]
            direction = factor.get("direction", "descending")

            if name not in df.columns:
                logger.warning(f"  因子 '{name}' 不在数据中（权重 {weight} 丢失）")
                continue

            # 处理缺失值
            col = df[name].copy()
            col = col.replace([np.inf, -np.inf], np.nan)

            # 检查是否全为默认值（全零数据无区分度，跳过）
            col_valid = col.dropna()
            if len(col_valid) == 0 or col_valid.nunique() <= 1:
                logger.debug(f"  因子 '{name}' 无变化（全 {col_valid.iloc[0] if len(col_valid) > 0 else 'NaN'}），跳过")
                continue

            if direction == "ascending":
                rank = col.rank(method="min", ascending=True, na_option="bottom")
            else:
                rank = col.rank(method="min", ascending=False, na_option="bottom")

            max_rank = rank.max()
            if max_rank > 0:
                normalized = (rank / max_rank) * 100
            else:
                normalized = pd.Series(50, index=rank.index)

            df["composite_score"] += normalized * weight
            total_weight += weight
            logger.debug(f"  因子 {name}: weight={weight}, direction={direction}, "
                        f"有效值={len(col_valid)}")

        # 归一化: 按有效权重等比缩放
        if total_weight > 0 and total_weight < 1.0:
            df["composite_score"] = df["composite_score"] / total_weight
            logger.info(f"  L4 有效总权重: {total_weight:.2f}, 评分已等比缩放")

        df["composite_score"] = df["composite_score"].round(2)

        # 诊断: 输出评分分布
        if len(df) > 0:
            scores = df["composite_score"]
            logger.info(f"  L4 评分分布: min={scores.min():.1f}, max={scores.max():.1f}, "
                        f"mean={scores.mean():.1f}, median={scores.median():.1f}")
        return df

    # ========================================
    #  v5 反弹引擎
    # ========================================

    def rebound_engine(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        v5 反弹引擎：识别超跌+放量反弹的个股。
        作为短线补充候选。
        """
        cfg = self.config.get("rebound_engine", {})
        if not cfg.get("enabled", False):
            return pd.DataFrame()

        lookback = cfg.get("lookback_days", 10)
        decline_threshold = cfg.get("decline_threshold", -0.08)
        volume_surge = cfg.get("volume_surge_ratio", 1.5)
        confirm_days = cfg.get("bounce_confirm_days", 2)

        if len(candidates) == 0:
            return pd.DataFrame()

        codes = [str(c).zfill(6) for c in (candidates["code"].tolist() if "code" in candidates.columns else [])]
        rebound_picks = []

        for code in codes:
            try:
                df = self.price_loader.get_price(code, days=lookback + 10)
                if len(df) < lookback + 5:
                    continue

                # 计算近N日跌幅
                close = df["close"].values
                decline = close[-1] / close[-1 - lookback] - 1

                if decline > decline_threshold:
                    continue  # 未达超跌阈值

                # 检查最近几天是否放量反弹
                recent_vol = df["volume"].tail(confirm_days).mean()
                prior_vol = df["volume"].iloc[-(confirm_days + 5):-confirm_days].mean()

                if prior_vol > 0 and recent_vol / prior_vol >= volume_surge:
                    recent_return = close[-1] / close[-1 - confirm_days] - 1
                    if recent_return > 0:  # 确认反弹
                        name = candidates[candidates["code"] == code]["name"].iloc[0] if "name" in candidates.columns else code
                        rebound_picks.append({
                            "code": code,
                            "name": name,
                            "decline_10d": round(decline * 100, 2),
                            "volume_ratio": round(recent_vol / prior_vol, 2),
                            "bounce_return": round(recent_return * 100, 2),
                            "sector": sector_of(code, self.sector_mapping),
                        })
            except Exception as e:
                logger.debug(f"反弹引擎处理 {code} 失败: {e}")

        result = pd.DataFrame(rebound_picks)
        if len(result) > 0:
            result = result.sort_values("volume_ratio", ascending=False)
            logger.info(f"v5 反弹引擎: 发现 {len(result)} 只超跌反弹候选")
        else:
            logger.info("v5 反弹引擎: 无符合条件的候选")

        return result

    # ========================================
    #  输出分类
    # ========================================

    def categorize(self, l4_results: pd.DataFrame, rebound_picks: pd.DataFrame,
                   holdings: dict) -> dict:
        """
        将结果分类为:
        ②A 质量榜, ②B 短线榜, ③A 持仓, ③B 操作建议, ③C 观察名单
        """
        out_cfg = self.config["output"]
        quality_n = out_cfg.get("quality_top_n", 10)
        short_n = out_cfg.get("short_term_top_n", 5)
        watch_n = out_cfg.get("watchlist_top_n", 23)

        # ②A 质量榜：综合评分最高的N只
        if len(l4_results) > 0:
            quality = l4_results.head(quality_n).copy()
        else:
            quality = pd.DataFrame()

        # ②B 短线榜：动量最强 或 反弹引擎选出的
        short_list = pd.DataFrame()
        if len(l4_results) > 0 and "momentum_20d" in l4_results.columns:
            short_list = l4_results.nlargest(short_n, "momentum_20d").copy()
        if len(rebound_picks) > 0:
            short_list = pd.concat([short_list, rebound_picks], ignore_index=True)

        # ③A 持仓：当前账户中的持仓
        holding_codes = list(holdings.keys()) if holdings else []
        if holding_codes and len(l4_results) > 0:
            holdings_df = l4_results[l4_results["code"].isin(holding_codes)].copy()
        else:
            holdings_df = pd.DataFrame()

        # ③B 操作建议：持仓中评分明显下降的（建议卖出/减仓）
        sell_list = pd.DataFrame()
        if len(holdings_df) > 0 and len(l4_results) > 0:
            # 持仓股如果不在 L4 前50%则建议关注
            median_score = l4_results["composite_score"].median()
            sell_list = holdings_df[holdings_df["composite_score"] < median_score].copy()

        # ③C 观察名单：评分靠前但未入质量榜的
        if len(l4_results) > quality_n:
            watchlist = l4_results.iloc[quality_n:quality_n + watch_n].copy()
        else:
            watchlist = pd.DataFrame()

        return {
            "②A_质量榜": quality,
            "②B_短线榜": short_list,
            "③A_持仓": holdings_df,
            "③B_操作建议": sell_list,
            "③C_观察名单": watchlist,
        }

    # ========================================
    #  ETF 选股
    # ========================================

    def select_etfs(self, top_n: int = 8) -> pd.DataFrame:
        """
        ETF 筛选：从主流行业/宽基 ETF 中按动量+流动性选 Top N。
        优先 DB 批量查询，回退 CLI。
        """
        well_known_etfs = [
            ("510050", "上证50ETF"), ("510300", "沪深300ETF"),
            ("510500", "中证500ETF"), ("159915", "创业板ETF"),
            ("588000", "科创50ETF"), ("512880", "证券ETF"),
            ("512100", "中证1000ETF"), ("159845", "中证1000"),
            ("512690", "酒ETF"), ("515790", "光伏ETF"),
            ("159995", "芯片ETF"), ("512480", "半导体ETF"),
            ("515050", "5GETF"), ("516510", "云计算ETF"),
        ]

        # 批量从 DB 获取 ETF 动量 + 成交额
        codes_only = [c for c, _ in well_known_etfs]
        db_momentum = self._batch_calc_momentum(codes_only)
        # 从 daily_price 取最近5日均成交额
        etf_amounts = {}
        try:
            mdb = get_market_db()
            batch_codes = [f"{c}.SZ" for c in codes_only] + [f"{c}.SH" for c in codes_only]
            placeholders = ",".join("?" for _ in batch_codes)
            rows = mdb.conn.execute(f"""
                SELECT code, amount FROM daily_price
                WHERE code IN ({placeholders}) AND amount > 0
                ORDER BY code, date DESC
                LIMIT {len(batch_codes) * 5}
            """, batch_codes).fetchall()
            from collections import defaultdict
            amt_by_code = defaultdict(list)
            for r in rows:
                code = r["code"].split(".")[0].zfill(6)
                amt_by_code[code].append(r["amount"])
            for code, amounts in amt_by_code.items():
                etf_amounts[code] = sum(amounts[:5]) / min(len(amounts), 5)
        except Exception:
            pass

        picks = []
        for code, name in well_known_etfs:
            try:
                # 优先 DB 动量
                if code in db_momentum:
                    mom20 = db_momentum[code].get("momentum_20d", 0)
                    mom60 = db_momentum[code].get("momentum_60d", 0)
                else:
                    # 回退 CLI
                    df = self.price_loader.get_price(code, days=60)
                    if len(df) < 20:
                        continue
                    close = df["close"].values
                    mom20 = (close[-1] / close[-20] - 1) * 100 if len(close) >= 21 else 0
                    mom60 = (close[-1] / close[-60] - 1) * 100 if len(close) >= 61 else 0

                # 综合评分: 动量(60%) + 流动性(40%)
                mom_score = (mom20 * 0.6 + (0 if not pd.notna(mom60) else mom60) * 0.4) * 0.6
                if not pd.notna(mom_score):
                    mom_score = 0
                score = mom_score + 20  # 基础分+动量分

                picks.append({
                    "code": code, "name": name,
                    "etf_type": "宽基" if code.startswith(("51","58")) else "行业",
                    "momentum_20d": round(mom20, 2) if pd.notna(mom20) else 0,
                    "momentum_60d": round(mom60, 2) if pd.notna(mom60) else 0,
                    "amount": etf_amounts.get(code, 0),
                    "score": round(score + 50, 1),
                })
            except Exception:
                continue

        result = pd.DataFrame(picks)
        if len(result) > 0:
            result = result.sort_values("score", ascending=False).head(top_n)
            logger.info(f"ETF 选股: {len(result)} 只, top={result.iloc[0].get('name','')}")
        return result

    def _get_local_stock_list(self) -> Optional[pd.DataFrame]:
        """
        从 market.db 本地快照构建股票列表（网络不可用时的降级方案）。
        快照已包含: code, name, sector, close, pe, pb, market_cap, change_pct 等。
        """
        try:
            df = self._load_local_snapshot()
        except Exception as e:
            logger.warning(f"加载本地快照异常: {type(e).__name__}: {e}")
            return None
        
        if df is None or len(df) == 0:
            logger.warning("本地快照为空，降级失败")
            return None
        
        # 检查并报告可用列
        required = ["code", "name", "close", "pe", "pb", "market_cap"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"本地快照缺少必选列: {missing}，可用列: {list(df.columns)[:15]}...")
            return None
        
        logger.info(f"使用本地快照作为股票列表: {len(df)} 只（离线模式），列: {list(df.columns)[:10]}...")
        return df

    # ========================================
    #  主运行入口
    # ========================================

    def run(self, save_to_db: bool = True, session_type: str = "pre_market") -> dict:
        """
        完整运行选股流程。
        Args:
            save_to_db: 是否保存结果到 SQLite（默认 True）
            session_type: 'pre_market' 或 'post_market'
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("A股多因子选股引擎启动")
        logger.info("=" * 60)

        # L0: 市场环境判断
        regime_info = self.assess_regime()

        # 获取全A股列表（网络 → 本地快照降级）
        stock_list = None
        try:
            stock_list = self.cli.get_stock_list()
            logger.info(f"获取股票列表: {len(stock_list)} 只")
        except Exception as e:
            logger.warning(f"网络股票列表失败: {e}，尝试本地快照降级...")
            stock_list = self._get_local_stock_list()

        if stock_list is None or len(stock_list) == 0:
            return {"error": "股票列表获取失败（网络和本地快照均不可用）", "regime": regime_info}

        # L2: 基础过滤
        filtered = self.filter_l2(stock_list)

        # L4: 多因子评分
        l4_results = self.score_l4(filtered)

        # v5 反弹引擎
        rebound_picks = self.rebound_engine(filtered)

        # 输出分类
        holdings = self.config.get("account", {}).get("holdings", {})
        categories = self.categorize(l4_results, rebound_picks, holdings)

        # ETF 选股
        etf_picks = self.select_etfs()

        # 概念板块归属增强
        concept_stats = pd.DataFrame()
        try:
            from tushare_provider import get_tushare
            ts = get_tushare()
            concept_stats = ts.get_concept_stats()
            if len(concept_stats) > 0 and len(l4_results) > 0:
                # 合并：给 l4_results 每只股票加上最热概念
                cs = concept_stats[["code", "concept_name", "concept_chg", "concept_amount"]]
                l4_results["code_str"] = l4_results["code"].astype(str).str.zfill(6)
                cs["code_str"] = cs["code"].astype(str).str.zfill(6)
                l4_results = l4_results.merge(
                    cs[["code_str", "concept_name", "concept_chg"]],
                    on="code_str", how="left"
                )
                l4_results.drop(columns=["code_str"], inplace=True)
                logger.info(f"概念板块增强: {l4_results['concept_name'].notna().sum()} 只")
        except Exception as e:
            logger.debug(f"概念板块增强跳过: {e}")

        elapsed = time.time() - start_time
        logger.info(f"选股引擎运行完成，耗时 {elapsed:.1f}s")

        results = {
            "regime": regime_info,
            "l2_filtered_count": len(filtered),
            "l4_results": l4_results,
            "rebound_picks": rebound_picks,
            "etf_picks": etf_picks,
            "categories": categories,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 保存到 SQLite
        if save_to_db:
            try:
                db = get_db()

                # 保存选股结果
                run_id = db.save_run_results(results, categories, session_type)
                results["run_id"] = run_id

                # 短线增强后保存因子评分
                enriched = enrich(l4_results, self.price_loader)
                db.save_factor_scores(run_id, enriched)

                # 保存持仓快照
                if holdings:
                    db.save_holdings_snapshot(holdings, l4_results)

                logger.info(f"数据已入库: run_id={run_id}")
            except Exception as e:
                logger.warning(f"数据入库失败（不影响选股结果）: {e}")

        return results


# ---- 命令行入口 ----
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data_cache/engine.log", encoding="utf-8"),
        ]
    )
    engine = MultiFactorEngine()
    results = engine.run()
    print(f"\n运行完成，耗时 {results['elapsed_seconds']}s")
    print(f"市场环境: {results['regime']['regime']} (仓位上限 {results['regime']['position_cap']:.0%})")
    for cat_name, cat_df in results["categories"].items():
        print(f"  {cat_name}: {len(cat_df)} 只")
