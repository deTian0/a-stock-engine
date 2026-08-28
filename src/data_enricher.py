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


def enrich_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    补全成交额(amount)与量比(volume_ratio)——流动性标签(liquidity_tag)与ETF成交额的前提列。

    回退链: 已有有效 amount/量比 → 保持; 缺失或<=0 → westock batch_quotes(实时, 含 amount/volume_ratio)。
    仅填充缺失/为0的行, 绝不覆盖已有有效值。
    """
    if len(df) == 0 or "code" not in df.columns:
        return df
    need_amt = ("amount" not in df.columns) or bool((df["amount"].isna() | (df["amount"] <= 0)).all())
    need_vr = ("volume_ratio" not in df.columns) or bool((df["volume_ratio"].isna() | (df["volume_ratio"] <= 0)).all())
    if not need_amt and not need_vr:
        return df
    codes = df["code"].astype(str).str.zfill(6).unique().tolist()
    try:
        from westock_helpers import batch_quotes
        q = batch_quotes(codes)
        if not q:
            logger.warning("amount/量比: westock返回空")
            return df
        if need_amt:
            if "amount" not in df.columns:
                df["amount"] = np.nan
            amap = df["code"].astype(str).str.zfill(6).map(lambda c: (q.get(c) or {}).get("amount"))
            mask = df["amount"].isna() | (df["amount"] <= 0)
            df.loc[mask, "amount"] = amap[mask]
        if need_vr:
            if "volume_ratio" not in df.columns:
                df["volume_ratio"] = np.nan
            vmap = df["code"].astype(str).str.zfill(6).map(lambda c: (q.get(c) or {}).get("volume_ratio"))
            mask = df["volume_ratio"].isna() | (df["volume_ratio"] <= 0)
            df.loc[mask, "volume_ratio"] = vmap[mask]
        logger.info(f"amount/量比补全: westock({len(q)})")
    except Exception as e:
        logger.warning(f"amount/量比补全异常: {e}")
    return df


def enrich_l4_results(l4_results: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    对 L4 选股结果补全缺失字段。回退链:
      close → tushare daily_basic → westock kline → akshare 实时
      name  → tushare stock_basic
      amount/量比 → westock batch_quotes（流动性标签与技术面量比的前提）
      概念   → tushare ths_index/member/daily
      技术面 → westock technical
    """
    if len(l4_results) == 0:
        return l4_results

    df = l4_results.copy()
    # 先行补全 amount/量比: 下游 enrich_risk_metrics 的 liquidity_tag 依赖 amount,
    # 否则整段跳过 -> 早盘简报"流动性"列恒为 '-'。
    df = enrich_amount(df)
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

        # Plan C: akshare 腾讯实时兜底（westock 失败后，逐只补全仍缺失的 close）
        still_missing = df["close"].isna() | (df["close"] <= 0)
        if still_missing.any():
            try:
                from westock_cli import get_cli
                cli = get_cli()
                filled = 0
                for c in df.loc[still_missing, "code"].astype(str).str.zfill(6).tolist():
                    try:
                        q = cli.get_realtime_quote(c)
                        if q.get("price"):
                            df.loc[df["code"].astype(str).str.zfill(6) == c, "close"] = q["price"]
                            filled += 1
                    except Exception:
                        continue
                if filled:
                    base = report.get("close", "")
                    report["close"] = f"{base}+akshare({filled})" if base else f"akshare({filled})"
            except Exception as e:
                logger.warning(f"当日股价 akshare 兜底异常: {e}")

    # ---- 3. 概念板块 ----
    need_concept = ("concept_name" not in df.columns or
                    df["concept_name"].isna().all() or
                    df["concept_name"].fillna("").eq("").all())
    if need_concept:
        try:
            from tushare_provider import get_tushare
            from database import get_db
            ts = get_tushare()

            def _build_concept_map(cs: pd.DataFrame) -> dict:
                """从 concept_stats 构建 {code: (name, chg)}，取每只股票最热概念；逐行防御，坏行跳过。"""
                cmap = {}
                if len(cs) == 0 or "code" not in cs.columns or "concept_name" not in cs.columns:
                    return cmap
                for _, r in cs.iterrows():
                    try:
                        c = str(r.get("code", "")).zfill(6)
                        if not c or c == "000000":
                            continue
                        name = r.get("concept_name", "-") or "-"
                        chg = r.get("concept_chg", 0)
                        chg = float(chg) if pd.notna(chg) else 0.0
                        prev = cmap.get(c)
                        if prev is None or abs(chg) > abs(float(prev[1])):
                            cmap[c] = (name, chg)
                    except Exception:
                        continue  # 跳过单只坏行, 不整段失败
                return cmap

            def _apply_concept(cm: dict):
                df["concept_name"] = df["code"].astype(str).str.zfill(6).map(
                    lambda c: cm.get(c, ("-", 0))[0])
                df["concept_chg"] = df["code"].astype(str).str.zfill(6).map(
                    lambda c: cm.get(c, ("-", 0))[1])

            def _clear_concept_cache():
                try:
                    db = get_db()
                    db.conn.execute(
                        "DELETE FROM market_data_cache WHERE data_type='concept_stats' "
                        "OR data_type='concept_members'")
                    db.conn.commit()
                except Exception:
                    pass

            # 回退链: 首次尝试 -> 空或异常 都清缓存重取一次
            # (修复: 旧逻辑仅"返回空"走重试分支, too many values to unpack 等异常被吞,
            #  概念补全静默失败; 现统一重试, 与"已 retry 成功"的实际表现对齐)
            cs = None
            for attempt in range(2):
                try:
                    cs = ts.get_concept_stats()
                    if cs is not None and len(cs) > 0 and "concept_name" in cs.columns:
                        break
                except Exception as e:
                    logger.warning(f"概念板块获取异常(attempt {attempt+1}): {e}")
                _clear_concept_cache()  # 失败 -> 清过期缓存后重试

            if cs is not None and len(cs) > 0 and "concept_name" in cs.columns:
                cm = _build_concept_map(cs)
                _apply_concept(cm)
                report["concept"] = f"tushare({df['concept_name'].notna().sum()})"
            else:
                report["concept"] = "FAILED"
        except Exception as e:
            logger.warning(f"概念补全异常(已跳过, 不影响其它字段): {e}")
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
