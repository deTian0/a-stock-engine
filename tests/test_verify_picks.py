"""verify_picks 单元测试：代码归一化 / 简报解析 / 验证报告（正常/边界/异常）。

重点: generate_verification_report 中「验证成功率」实为**数据可用率**（status==success 占比），
而非胜率（正收益占比）——标签存在语义误导（见下方文档化断言）。
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


def test_generate_report_success_rate_equals_success_over_total():
    # 文档化「验证成功率」的语义: success/total, 而非胜率
    vr = [_vr_with_returns([5.0, -3.0]), _vr_with_returns([None])]
    rep = vp.generate_verification_report(vr)
    # 2 成功 / 3 总 = 66.7% 验证成功率
    assert "验证成功率" in rep
    assert "66.7%" in rep.split("验证成功率")[1].split("\n")[0]


def test_generate_report_empty():
    rep = vp.generate_verification_report([{"date": "2026-08-26", "error": "无数据", "picks": []}])
    assert "无数据" in rep  # generate_verification_report 原样回显传入的 error 字段
