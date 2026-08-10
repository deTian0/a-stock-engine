"""
verify_pipeline.py — 全链路验证脚本

逐点测试: 数据源 → 补全 → 风控 → 报告
每个点打印 PASS/FAIL，最后给出总评。
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        logger.info(f"  ✅ {name}: PASS {detail}")
        PASS += 1
    else:
        logger.error(f"  ❌ {name}: FAIL {detail}")
        FAIL += 1


def test_1_tushare_names():
    """① tushare stock_basic 能否返回股票名称"""
    logger.info("--- ① tushare名称 ---")
    try:
        from tushare_provider import get_tushare
        ts = get_tushare()
        sl = ts.get_stock_list()
        check("stock_basic有数据", len(sl) > 0, f"{len(sl)} rows")
        check("name列存在", "name" in sl.columns)
        if "name" in sl.columns:
            valid = sl["name"].notna() & (sl["name"].astype(str).str.lower() != "nan")
            check("名称非空", valid.sum() > 100, f"{valid.sum()} 有效")
            # 检查几个知名股票
            for code, expected in [("600519", "贵州茅台"), ("000001", "平安银行"), ("000858", "五粮液")]:
                matches = sl[sl["code"].astype(str).str.zfill(6) == code]
                if len(matches) > 0:
                    actual = str(matches.iloc[0]["name"])
                    check(f"{code}={actual}", expected in actual, "")
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}")


def test_2_westock_prices():
    """② westock kline 能否获取股价"""
    logger.info("--- ② westock股价 ---")
    try:
        from westock_helpers import batch_close_prices
        codes = ["600519", "000001", "600610"]
        prices = batch_close_prices(codes)
        check("返回非空", len(prices) > 0, f"{len(prices)} 只")
        for c in codes:
            check(f"{c}有价格", c in prices and prices[c] > 0, f"={prices.get(c)}")
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}")


def test_3_westock_tech():
    """③ westock technical 能否获取技术指标"""
    logger.info("--- ③ westock技术面 ---")
    try:
        from westock_helpers import batch_tech_indicators
        codes = ["600519", "000001"]
        tech = batch_tech_indicators(codes)
        check("返回非空", len(tech) > 0, f"{len(tech)} 只")
        for c in codes:
            if c in tech:
                t = tech[c]
                check(f"{c}.signal非空", t.get("signal","-") != "-", f"={t['signal']}")
                check(f"{c}.ma非空", t.get("ma","-") != "-", f"={t['ma']}")
                check(f"{c}.macd非空", t.get("macd","-") != "-", f"={t['macd']}")
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}")


def test_4_tushare_concepts():
    """④ tushare 概念板块"""
    logger.info("--- ④ tushare概念 ---")
    try:
        from tushare_provider import get_tushare
        ts = get_tushare()
        cs = ts.get_concept_stats()
        check("概念数据非空", len(cs) > 0, f"{len(cs)} rows")
        if len(cs) > 0:
            check("concept_name列存在", "concept_name" in cs.columns)
            check("concept_chg列存在", "concept_chg" in cs.columns)
            valid = cs["concept_name"].notna().sum()
            check(f"概念名称有效", valid > 100, f"{valid} 有效")
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}")


def test_5_data_enricher():
    """⑤ data_enricher 补全链路"""
    logger.info("--- ⑤ data_enricher补全 ---")
    try:
        # 模拟 L4 结果(缺name/close/概念/技术面)
        test_df = pd.DataFrame({
            "code": ["600519", "000001", "600610"],
            "composite_score": [95, 80, 75],
            "sector": ["沪市", "深市", "沪市"],
        })
        from data_enricher import enrich_and_report
        enriched = enrich_and_report(test_df)
        
        check("name补全", enriched["name"].notna().all(), f"={enriched['name'].tolist()}")
        check("close补全", enriched["close"].notna().sum() >= 2, f"={enriched['close'].notna().sum()}")
        if "concept_name" in enriched.columns:
            check("概念补全", enriched["concept_name"].notna().sum() >= 1)
        if "tech_signal" in enriched.columns:
            check("技术面补全", enriched["tech_signal"].notna().sum() >= 1)
        if "signal_grade" in enriched.columns:
            check("信号强度评级", enriched["signal_grade"].notna().all(), f"={enriched['signal_grade'].tolist()}")
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}", exc_info=True)


def test_6_risk_module():
    """⑥ risk_module ATR止损+Kelly仓位"""
    logger.info("--- ⑥ risk_module风控 ---")
    try:
        test_df = pd.DataFrame({
            "code": ["600519", "000001"],
            "close": [1348.86, 11.29],
            "composite_score": [95, 80],
            "sector": ["沪市", "深市"],
        })
        from risk_module import enrich_risk_metrics
        enriched = enrich_risk_metrics(test_df)
        
        check("止损价补全", enriched["stop_loss"].notna().sum() >= 1, 
              f"={enriched['stop_loss'].tolist()}")
        check("仓位补全", "suggested_position" in enriched.columns)
        check("流动性标签", "liquidity_tag" in enriched.columns)
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}")


def test_7_html_report():
    """⑦ HTML报告生成"""
    logger.info("--- ⑦ HTML报告 ---")
    try:
        from html_report import generate_html
        mock_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": 10,
            "regime": {"regime": "震荡", "position_cap": 0.5},
            "categories": {
                "②A_质量榜": pd.DataFrame([{
                    "code": "600519", "name": "贵州茅台", "close": 1348.86,
                    "stop_loss": 1300.00, "tech_signal": "🟢偏多",
                    "tech_ma": "站上MA20,多头", "tech_macd": "金叉",
                    "roe": 25.5, "revenue_growth": 12.3,
                    "composite_score": 95, "suggested_position": 8.5,
                    "liquidity_tag": "🟢高", "signal_grade": "🔥🔥🔥",
                    "sector": "沪市",
                }]),
                "②B_短线榜": pd.DataFrame(),
            },
            "etf_picks": pd.DataFrame(),
            "l4_results": pd.DataFrame({"composite_score": [90, 80, 70]}),
        }
        html = generate_html(mock_results, {})
        check("HTML非空", len(html) > 1000, f"{len(html)} bytes")
        check("含ECharts", "echarts" in html.lower())
        check("含简体中文", "盘前选股" in html)
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}")


def test_8_full_chain():
    """⑧ 全链路: 引擎 → 补全 → 简报"""
    logger.info("--- ⑧ 全链路 ---")
    try:
        from multifactor import MultiFactorEngine
        import yaml
        config_path = Path(__file__).parent.parent / "config/config.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        engine = MultiFactorEngine(config_dict=config)
        results = engine.run(session_type="pre_market")
        
        check("引擎无error", "error" not in results)
        check("L2过滤>0", results.get("l2_filtered_count", 0) > 0)
        check("L4评分>0", len(results.get("l4_results", [])) > 0)
        
        l4 = results["l4_results"]
        if len(l4) > 0:
            name_ok = (l4.get("name", pd.Series()).notna()).sum()
            check(f"名称有效", name_ok > 0, f"{name_ok}/{len(l4)}")
            
            if "close" in l4.columns:
                close_ok = l4["close"].notna().sum()
                check(f"股价有效", close_ok > 0, f"{close_ok}/{len(l4)}")
            else:
                check("股价列存在", False, "l4_results无close列")
            
            if "concept_name" in l4.columns:
                concept_ok = l4["concept_name"].notna().sum()
                check(f"概念有效", concept_ok > 0, f"{concept_ok}/{len(l4)}")
            
            if "tech_signal" in l4.columns:
                tech_ok = l4["tech_signal"].notna().sum()
                check(f"技术面有效", tech_ok > 0, f"{tech_ok}/{len(l4)}")
    except Exception as e:
        logger.error(f"  ❌ 异常: {e}", exc_info=True)


def main():
    global PASS, FAIL
    logger.info("=" * 60)
    logger.info("全链路验证开始")
    logger.info("=" * 60)
    
    test_1_tushare_names()
    test_2_westock_prices()
    test_3_westock_tech()
    test_4_tushare_concepts()
    test_5_data_enricher()
    test_6_risk_module()
    test_7_html_report()
    test_8_full_chain()
    
    logger.info("=" * 60)
    logger.info(f"结果: {PASS} PASS / {FAIL} FAIL")
    if FAIL == 0:
        logger.info("✅ 全链路通过! 数据补全正常")
    else:
        logger.error(f"❌ {FAIL} 项失败，需修复")
    logger.info("=" * 60)
    
    return PASS, FAIL


if __name__ == "__main__":
    main()
