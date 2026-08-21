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
FAILED_ITEMS = []   # 汇总所有未通过项: {group, name, detail, kind}
CURRENT_GROUP = ""  # 当前测试组名（由 main 在调用各 test_* 前设置）

# 各测试组对应的「修复建议」——方便定位要改的数据/配置
GROUP_HINTS = {
    "① tushare名称": "数据源 Tushare：检查 .env 中的 TUSHARE_TOKEN 是否有效、网络能否出网（火绒可能拦截 Python 出网）、stock_basic 接口返回。",
    "② westock股价": "数据源 westock-data：检查 westock CLI 是否安装可用、Node 环境是否正常、batch_close_prices 返回。",
    "③ westock技术面": "数据源 westock-data：检查 westock_helpers.batch_tech_indicators 与 CLI 技术指标返回。",
    "④ tushare概念": "数据源 Tushare：检查 get_concept_stats 接口、TUSHARE_TOKEN 积分是否足够（概念板块需 5000+ 积分）。",
    "⑤ data_enricher补全": "补全链路：检查 data_enricher.enrich_and_report 中 名称/股价/概念/技术面 的回填逻辑与上游数据。",
    "⑥ risk_module风控": "风控模块：检查 risk_module.enrich_risk_metrics 的 ATR 止损价/Kelly 仓位/流动性标签计算。",
    "⑦ HTML报告": "报告渲染：检查 html_report.generate_html 的模板与传入的 mock 数据结构。",
    "⑧ 全链路": "引擎全链路：检查 multifactor.MultiFactorEngine.run 的 L2/L4 过滤与各因子数据源回填。",
}


def record_exception(e, exc_info=False):
    """记录一次测试级异常到未通过清单（同时计入 FAIL）。"""
    global FAIL
    FAIL += 1
    FAILED_ITEMS.append({
        "group": CURRENT_GROUP,
        "name": "测试抛出异常",
        "detail": str(e),
        "kind": "exception",
    })
    if exc_info:
        logger.error(f"  ❌ 测试异常: {e}", exc_info=True)
    else:
        logger.error(f"  ❌ 测试异常: {e}")


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        logger.info(f"  ✅ {name}: PASS {detail}")
        PASS += 1
    else:
        logger.error(f"  ❌ {name}: FAIL {detail}")
        FAIL += 1
        FAILED_ITEMS.append({
            "group": CURRENT_GROUP,
            "name": name,
            "detail": detail,
            "kind": "check",
        })


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
        record_exception(e)


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
        record_exception(e)


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
        record_exception(e)


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
        record_exception(e)


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
        record_exception(e, exc_info=True)


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
        record_exception(e)


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
        record_exception(e)


def test_8_full_chain():
    """⑧ 全链路: 引擎 → 补全 → 简报"""
    logger.info("--- ⑧ 全链路 ---")
    try:
        from multifactor import MultiFactorEngine
        import yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
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
        record_exception(e, exc_info=True)


def main():
    global PASS, FAIL, CURRENT_GROUP
    logger.info("=" * 60)
    logger.info("全链路验证开始")
    logger.info("=" * 60)

    test_plan = [
        ("① tushare名称", test_1_tushare_names),
        ("② westock股价", test_2_westock_prices),
        ("③ westock技术面", test_3_westock_tech),
        ("④ tushare概念", test_4_tushare_concepts),
        ("⑤ data_enricher补全", test_5_data_enricher),
        ("⑥ risk_module风控", test_6_risk_module),
        ("⑦ HTML报告", test_7_html_report),
        ("⑧ 全链路", test_8_full_chain),
    ]
    for group, fn in test_plan:
        CURRENT_GROUP = group
        try:
            fn()
        except Exception as e:
            record_exception(e, exc_info=True)

    # ===== 末尾汇总：未通过项清单 =====
    logger.info("=" * 60)
    logger.info(f"结果: {PASS} PASS / {FAIL} FAIL")
    if FAILED_ITEMS:
        logger.error("=" * 60)
        logger.error(f"❌ 共 {FAIL} 项未通过，请按下列清单针对性修改相关数据/配置：")
        for idx, item in enumerate(FAILED_ITEMS, 1):
            if item["kind"] == "check":
                suffix = f"  → {item['detail']}" if item["detail"] else ""
                logger.error(f"  {idx:>2}. [{item['group']}] {item['name']}{suffix}")
            else:
                logger.error(f"  {idx:>2}. [{item['group']}] 异常: {item['detail']}")
        logger.error("-" * 60)
        logger.error("修复建议（仅列出失败项所属模块）：")
        for group, hint in GROUP_HINTS.items():
            if any(i["group"] == group for i in FAILED_ITEMS):
                logger.error(f"  [{group}] {hint}")
    else:
        logger.info("✅ 全链路通过! 数据补全正常")
    logger.info("=" * 60)

    return PASS, FAIL


if __name__ == "__main__":
    main()
