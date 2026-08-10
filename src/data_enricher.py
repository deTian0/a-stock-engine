"""
data_enricher.py — 统一数据补全模块

借鉴 vnpy gateway 模式: 单一入口、多源回退、失败记warning。
确保L4选股结果中 name/close/概念/技术面 100%填充。
"""

import logging
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def enrich_l4_results(l4_results: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    对 L4 选股结果补全缺失字段。回退链:
      close → tushare daily_basic → westock kline
      name  → tushare stock_basic
      概念   → tushare ths_index/member/daily
      技术面 → westock technical
    """
    if len(l4_results) == 0:
        return l4_results

    df = l4_results.copy()
    codes = df["code"].astype(str).str.zfill(6).unique().tolist()
    report = {}

    # ---- 1. 名称 ----
    if "name" not in df.columns or df["name"].isna().all() or (df["name"].astype(str).str.lower() == "nan").all():
        try:
            from tushare_provider import get_tushare
            ts = get_tushare()
            sl = ts.get_stock_list()
            if "code" in sl.columns and "name" in sl.columns:
                nm = dict(zip(sl["code"].astype(str).str.zfill(6), sl["name"]))
                df["name"] = df["code"].astype(str).str.zfill(6).map(nm).fillna(df["code"])
                report["name"] = f"tushare({df['name'].notna().sum()})"
        except Exception as e:
            logger.warning(f"名称补全失败: {e}")
            df["name"] = df["name"].fillna(df["code"])
            report["name"] = "fallback(code)"

    # ---- 2. 当日股价 (close) ----
    need_close = "close" not in df.columns or df["close"].isna().all() or (df["close"] <= 0).all()
    if need_close:
        # Plan A: tushare daily_basic 已有 close
        if "close" in df.columns and df["close"].notna().sum() > 0:
            report["close"] = f"tushare({df['close'].notna().sum()})"
        else:
            # Plan B: westock kline
            try:
                from westock_helpers import batch_close_prices
                cm = batch_close_prices(codes)
                if cm:
                    df["close"] = df["code"].astype(str).str.zfill(6).map(cm)
                    report["close"] = f"westock({pd.notna(df['close']).sum()})"
                else:
                    logger.warning("当日股价: westock返回空")
                    report["close"] = "FAILED"
            except Exception as e:
                logger.warning(f"当日股价补全异常: {e}")
                report["close"] = f"ERROR:{e}"

    # ---- 3. 概念板块 ----
    need_concept = ("concept_name" not in df.columns or 
                    df["concept_name"].isna().all() or 
                    df["concept_name"].fillna("").eq("").all())
    if need_concept:
        try:
            from tushare_provider import get_tushare
            ts = get_tushare()
            cs = ts.get_concept_stats()
            if len(cs) > 0 and "code" in cs.columns and "concept_name" in cs.columns:
                concept_map = {}
                for _, r in cs.iterrows():
                    c = str(r["code"]).zfill(6)
                    if c not in concept_map or abs(r.get("concept_chg", 0)) > abs(concept_map.get(c, (0,))[1]):
                        concept_map[c] = (r["concept_name"], r.get("concept_chg", 0))
                
                df["concept_name"] = df["code"].astype(str).str.zfill(6).map(
                    lambda c: concept_map.get(c, ("-", 0))[0])
                df["concept_chg"] = df["code"].astype(str).str.zfill(6).map(
                    lambda c: concept_map.get(c, ("-", 0))[1])
                report["concept"] = f"tushare({df['concept_name'].notna().sum()})"
            else:
                logger.warning("概念板块: tushare返回空, 缓存可能过期, 删除缓存重试")
                # 清缓存重试
                from database import get_db
                db = get_db()
                db.conn.execute("DELETE FROM market_data_cache WHERE data_type='concept_stats' OR data_type='concept_members'")
                db.conn.commit()
                cs2 = ts.get_concept_stats()
                if len(cs2) > 0:
                    concept_map = {}
                    for _, r in cs2.iterrows():
                        c = str(r["code"]).zfill(6)
                        if c not in concept_map:
                            concept_map[c] = (r["concept_name"], r.get("concept_chg", 0))
                    df["concept_name"] = df["code"].astype(str).str.zfill(6).map(lambda c: concept_map.get(c, ("-", 0))[0])
                    df["concept_chg"] = df["code"].astype(str).str.zfill(6).map(lambda c: concept_map.get(c, ("-", 0))[1])
                    report["concept"] = f"tushare(retry:{df['concept_name'].notna().sum()})"
                else:
                    report["concept"] = "FAILED"
        except Exception as e:
            logger.warning(f"概念补全异常: {e}")
            report["concept"] = f"ERROR:{e}"

    # ---- 4. 技术面信号 ----
    need_tech = "tech_signal" not in df.columns
    if need_tech:
        try:
            from westock_helpers import batch_tech_indicators
            tech = batch_tech_indicators(codes)
            if tech:
                for k in ["signal", "ma", "macd", "rsi"]:
                    df[f"tech_{k}"] = df["code"].astype(str).str.zfill(6).map(
                        lambda c, k=k: tech.get(c, {}).get(k, "-"))
                report["tech"] = f"westock({len(tech)})"
            else:
                logger.warning("技术面: westock technical 返回空")
                report["tech"] = "FAILED"
        except Exception as e:
            logger.warning(f"技术面补全异常: {e}")
            report["tech"] = f"ERROR:{e}"

    # ---- 5. 补 sector 回退 ----
    if "sector" not in df.columns or df["sector"].isna().all():
        df["sector"] = df["code"].astype(str).apply(_infer_sector)

    logger.info(f"数据补全完成: {report}")
    return df


def grade_signal(row) -> str:
    """
    Qbot 风格信号强度评级。
    综合 技术面(偏多/震荡/偏空) + 基本面(ROE) + 评分 → 🔥🔥🔥/🔥🔥/🔥/⚪
    """
    score = row.get("composite_score", 50)
    tech = str(row.get("tech_signal", ""))
    roe = row.get("roe", 0) if pd.notna(row.get("roe")) else 0

    bullish_bonus = 1 if "偏多" in tech else 0
    bearish_penalty = 1 if "偏空" in tech else 0
    roe_bonus = 1 if roe > 10 else (0.5 if roe > 5 else 0)

    effective = score / 100 + bullish_bonus * 0.15 + roe_bonus * 0.1 - bearish_penalty * 0.2

    if effective > 0.85:
        return "🔥🔥🔥"
    elif effective > 0.7:
        return "🔥🔥"
    elif effective > 0.55:
        return "🔥"
    else:
        return "⚪"


def _infer_sector(code: str) -> str:
    """代码前缀推断板块。"""
    code = str(code).zfill(6)
    m = {
        "60": "沪市", "68": "科创板", "00": "深市主板",
        "30": "创业板", "15": "ETF深", "51": "ETF沪",
        "56": "ETF沪", "58": "ETF沪",
    }
    for prefix, name in m.items():
        if code.startswith(prefix):
            return name
    return "未知"


def enrich_and_report(l4_results: pd.DataFrame) -> pd.DataFrame:
    """补全 + 报告。返回富化后的 DataFrame。"""
    before = {
        "name": (l4_results.get("name", pd.Series()).notna().sum() if "name" in l4_results.columns else 0),
        "close": (l4_results.get("close", pd.Series()).notna().sum() if "close" in l4_results.columns else 0),
        "concept": (l4_results.get("concept_name", pd.Series()).notna().sum() if "concept_name" in l4_results.columns else 0),
        "tech": (1 if "tech_signal" in l4_results.columns else 0),
    }
    result = enrich_l4_results(l4_results)
    # 信号强度评级
    result["signal_grade"] = result.apply(grade_signal, axis=1)
    # 确保所有列都存在（补全失败也不崩）
    for col in ["close", "concept_name", "concept_chg", "tech_signal", "tech_ma", "tech_macd", "tech_rsi"]:
        if col not in result.columns:
            result[col] = "-"
    after = {
        "name": result["name"].notna().sum(),
        "close": result["close"].notna().sum() if "close" in result.columns else 0,
        "concept": result["concept_name"].notna().sum() if "concept_name" in result.columns else 0,
        "tech": (result["tech_signal"].notna().sum() if "tech_signal" in result.columns else 0),
    }
    logger.info(f"data_enricher: name {before['name']}→{after['name']}, "
                f"close {before['close']}→{after['close']}, "
                f"concept {before['concept']}→{after['concept']}, "
                f"tech {before['tech']}→{after['tech']}")
    return result
