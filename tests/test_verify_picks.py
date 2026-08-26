"""verify_picks 单元测试：代码归一化 / 简报解析 / 验证报告（正常/边界/异常）。

重点: generate_verification_report 中「数据齐备率」为 T+2 价格可获取占比,
      与「胜率」（正收益占比）已明确区分, 不再混用「验证成功率」误导标签
      （见 test_generate_report_data_completeness_rate）。简报解析按列名定位
      （见 test_load_picks_from_brief_column_order_insensitive）。
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import verify_picks as vp


def test_norm_code_zfill_and_strip_suffix():
    assert vp.norm_code("6") == "000006"
    assert vp.norm_code("600519") == "600519"
    assert vp.norm_code("600519.SZ") == "600519"
    assert vp.norm_code(" 159852 ") == "159852"


def test_norm_code_non_digit_passthrough():
    assert vp.norm_code("abc") == "abc"


def test_load_picks_from_brief_only_buy_sections(tmp_path):
    md = tmp_path / "盘前选股简报.md"
    md.write_text(
        "# 标题\n"
        "## 一、中长线组合（建议持仓 5-20 日）\n"
        "| 代码 | 名称 | 股价 |\n"
        "|------|------|------|\n"
        "| 600519 | 茅台 | 1800 |\n"
        "## 三、当前持仓追踪\n"  # 非买入段, 应排除
        "| 代码 | 名称 | 当日股价 |\n"
        "|------|------|----------|\n"
        "| 159852 | 软件ETF | 0.66 |\n",
        encoding="utf-8",
    )
    picks = vp.load_picks_from_brief(md)
    codes = [p["code"] for p in picks]
    assert "600519" in codes
    assert "159852" not in codes           # 持仓追踪被排除
    assert picks[0]["category"] == "②A_质量榜"


def test_load_picks_from_brief_missing_file(tmp_path):
    assert vp.load_picks_from_brief(tmp_path / "nope.md") == []


def _vr_with_returns(rets):
    picks = []
    for i, r in enumerate(rets):
        if r is None:
            picks.append({"code": f"{i:06d}", "name": f"S{i}", "category": "②A_质量榜",
                          "status": "数据不足"})
        else:
            picks.append({"code": f"{i:06d}", "name": f"S{i}", "category": "②A_质量榜",
                          "status": "success", "t0_close": 10, "t2_close": 10 * (1 + r / 100),
                          "return_pct": round(r, 2)})
    return {"date": "2026-08-26", "picks": picks}


def test_generate_report_win_rate_correct():
    # 3 成功(2 正 1 负) + 1 数据不足
    vr = [
        _vr_with_returns([5.0, -3.0, 10.0]),
        _vr_with_returns([None]),
    ]
    rep = vp.generate_verification_report(vr)
    assert "胜率" in rep
    # 正收益 2 / 成功 3 = 66.7%
    assert "66.7%" in rep
    assert "平均涨幅" in rep


def test_generate_report_data_completeness_rate():
    # 「数据齐备率」语义: success/total（价格可获取占比）, 已与「胜率」区分
    vr = [_vr_with_returns([5.0, -3.0]), _vr_with_returns([None])]
    rep = vp.generate_verification_report(vr)
    # 2 成功 / 3 总 = 66.7% 数据齐备率
    assert "数据齐备率" in rep
    assert "验证成功率" not in rep          # 误导标签已移除
    assert "66.7%" in rep.split("数据齐备率")[1].split("\n")[0]


def test_generate_report_empty():
    rep = vp.generate_verification_report([{"date": "2026-08-26", "error": "无数据", "picks": []}])
    assert "无数据" in rep  # generate_verification_report 原样回显传入的 error 字段
