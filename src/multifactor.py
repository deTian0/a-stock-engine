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
from database import get_db
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
        """从单行基本面数据提取因子值。"""
        row = fund_row
        return {
            "pe": float(row.get("pe", 0)) if pd.notna(row.get("pe")) else 999,
            "pb": float(row.get("pb", 0)) if pd.notna(row.get("pb")) else 999,
            "roe": float(row.get("roe", 0)) if pd.notna(row.get("roe")) else 0,
            "gross_margin": float(row.get("gross_margin", 0)) if pd.notna(row.get("gross_margin")) else 0,
            "debt_ratio": float(row.get("debt_ratio", 100)) if pd.notna(row.get("debt_ratio")) else 100,
            "revenue_growth": float(row.get("revenue_growth", 0)) if pd.notna(row.get("revenue_growth")) else 0,
            "profit_growth": float(row.get("profit_growth", 0)) if pd.notna(row.get("profit_growth")) else 0,
            "market_cap": float(row.get("market_cap", 0)) if pd.notna(row.get("market_cap")) else 0,
            "name": str(row.get("name", "")) if "name" in row else "",
        }

    def _fetch_stock_data(self, code: str, fund_lookup: dict, sector_map: dict) -> dict:
        """获取单只股票的数据（价格+基本面+板块），用于并发调用。"""
        factor_values = {"code": code}

        # 从预构建的基本面字典中提取
        if code in fund_lookup:
            factor_values.update(fund_lookup[code])

        # 获取价格数据计算动量
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

        # 板块信息
        factor_values["sector"] = sector_map.get(code, "未知")

        return factor_values

    #  L4: 多因子评分
    # ========================================

    def score_l4(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        L4 多因子评分：对通过 L2 的股票进行综合评分排序。

        评分流程:
        1. 批量获取基本面数据
        2. 并发获取所有股票价格数据（ThreadPoolExecutor, 8 workers）
        3. 计算各因子值 + 加权综合评分
        4. 按评分降序排列
        """
        cfg = self.config["factor_l4"]
        factors_cfg = cfg["factors"]
        top_n = cfg.get("top_n", 30)

        if len(candidates) == 0:
            logger.warning("L4 无候选股票")
            return pd.DataFrame()

        codes = candidates["code"].tolist() if "code" in candidates.columns else []
        total = len(codes)
        logger.info(f"L4 评分: 开始处理 {total} 只股票...")

        # Step 1: 批量获取基本面数据
        try:
            fundamentals = self.cli.get_fundamentals(codes)
            logger.info(f"  基本面数据获取完成: {len(fundamentals)} 条")
        except Exception as e:
            logger.error(f"获取基本面数据失败: {e}")
            fundamentals = pd.DataFrame()

        # 构建基本面查找字典
        fund_lookup = {}
        if len(fundamentals) > 0 and "code" in fundamentals.columns:
            for _, row in fundamentals.iterrows():
                code = str(row.get("code", ""))
                fund_lookup[code] = self._extract_fundamentals(row)

        # Step 2: 预加载板块映射（避免并发中重复CLI调用）
        sector_map = self.sector_mapping

        # Step 3: 并发获取价格数据（信号安全）
        max_workers = min(self.config.get("concurrency", {}).get("max_workers", 4), total)
        logger.info(f"  并发获取价格数据: {max_workers} workers, {total} 只股票...")

        factor_list = []
        completed = 0
        futures_map = {}

        executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            # Reset shutdown flag before starting
            _executor_shutdown.clear()

            for code in codes:
                future = executor.submit(
                    self._fetch_stock_data, code, fund_lookup, sector_map
                )
                futures_map[future] = code
                # 错峰提交：每提交一个 worker 休息几毫秒，避免瞬时 CPU 尖峰
                if len(futures_map) <= max_workers:
                    delay = self.config.get("concurrency", {}).get("fetch_delay_ms", 50) / 1000.0
                    if delay > 0:
                        time.sleep(delay)

            for future in as_completed(futures_map):
                # 检查是否收到中断信号
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
                if completed % 100 == 0 or completed == total:
                    logger.info(f"  L4 进度: {completed}/{total}")

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt: 取消所有剩余任务...")
            for f in futures_map:
                f.cancel()
            executor.shutdown(wait=False)
            raise

        finally:
            # Ensure executor is always shutdown
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python <3.9 doesn't support cancel_futures
                for f in list(futures_map.keys()):
                    f.cancel()
                executor.shutdown(wait=False)

        # Step 4: 计算综合评分
        scores_df = pd.DataFrame(factor_list)
        scores_df = self._calc_composite_score(scores_df, factors_cfg)

        # 按评分降序排列，取前N
        scores_df = scores_df.sort_values("composite_score", ascending=False).head(top_n)

        logger.info(f"L4 评分完成: {len(scores_df)} 只入选 (top {top_n})")
        return scores_df.reset_index(drop=True)

    def _calc_composite_score(self, df: pd.DataFrame, factors_cfg: list) -> pd.DataFrame:
        """计算加权综合评分。"""
        df = df.copy()
        df["composite_score"] = 0.0

        for factor in factors_cfg:
            name = factor["name"]
            weight = factor["weight"]
            direction = factor.get("direction", "descending")

            if name not in df.columns:
                logger.warning(f"因子 {name} 不在数据中，跳过")
                continue

            # 处理缺失值
            col = df[name].copy()
            col = col.replace([np.inf, -np.inf], np.nan)

            if direction == "ascending":
                # 越低越好 → 排名反转
                rank = col.rank(method="min", ascending=True, na_option="bottom")
            else:
                # 越高越好
                rank = col.rank(method="min", ascending=False, na_option="bottom")

            # 归一化到 0-100
            max_rank = rank.max()
            if max_rank > 0:
                normalized = (rank / max_rank) * 100
            else:
                normalized = pd.Series(50, index=rank.index)

            df["composite_score"] += normalized * weight
            logger.debug(f"  因子 {name}: weight={weight}, direction={direction}")

        df["composite_score"] = df["composite_score"].round(2)
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

        codes = candidates["code"].tolist() if "code" in candidates.columns else []
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

        # 获取全A股列表
        try:
            stock_list = self.cli.get_stock_list()
            logger.info(f"获取股票列表: {len(stock_list)} 只")
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return {"error": str(e), "regime": regime_info}

        # L2: 基础过滤
        filtered = self.filter_l2(stock_list)

        # L4: 多因子评分
        l4_results = self.score_l4(filtered)

        # v5 反弹引擎
        rebound_picks = self.rebound_engine(filtered)

        # 输出分类
        holdings = self.config.get("account", {}).get("holdings", {})
        categories = self.categorize(l4_results, rebound_picks, holdings)

        elapsed = time.time() - start_time
        logger.info(f"选股引擎运行完成，耗时 {elapsed:.1f}s")

        results = {
            "regime": regime_info,
            "l2_filtered_count": len(filtered),
            "l4_results": l4_results,
            "rebound_picks": rebound_picks,
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
