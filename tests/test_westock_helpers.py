"""westock_helpers 单元测试：_to_ws（纯）与 batch_kline/batch_quotes（mock 子进程）。

覆盖: 正常解析 / 空输出 / 坏行跳过（异常分支）/ 不同市场前缀。
"""
import sys
from pathlib import Path

import subprocess
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import westock_helpers
from conftest import make_kline_stdout, fake_subprocess_run_factory


def test_to_ws_sh():
    assert westock_helpers._to_ws("600519") == "sh600519"


def test_to_ws_sz():
    assert westock_helpers._to_ws("000001") == "sz000001"


def test_to_ws_bj():
    assert westock_helpers._to_ws("889999") == "bj889999"


def test_to_ws_zfill():
    assert westock_helpers._to_ws("9852") == "sz009852"


def test_to_ws_etf_routing():
    # A股 ETF/基金代码必须正确路由(修复前 159xxx/51xxx 被错判为 bj)
    assert westock_helpers._to_ws("159852") == "sz159852"
    assert westock_helpers._to_ws("515790") == "sh515790"
    assert westock_helpers._to_ws("159552") == "sz159552"


def test_batch_kline_parses(monkeypatch):
    out = make_kline_stdout("sh600519", [10.0 - i * 0.1 for i in range(20)])
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run_factory(out))
    res = westock_helpers.batch_kline(["600519"], limit=20)
    assert "600519" in res
    assert len(res["600519"]) == 20
    assert res["600519"][0] == 10.0


def test_batch_kline_empty_on_no_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run_factory(""))
    assert westock_helpers.batch_kline(["600519"]) == {}


def test_batch_quotes_parses(monkeypatch):
    stdout = (
        "| symbol | date | open | last | high | low | volume | amount | exchange |\n"
        "| sh159852 | 2026-08-25 | 0.66 | 0.663 | 0.67 | 0.65 | 1000000 | 5000000 | 1.2 |\n"
        "| sz159869 | 2026-08-25 | 1.07 | 1.079 | 1.09 | 1.06 | 2000000 | 6000000 | 1.1 |\n"
    )
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run_factory(stdout))
    res = westock_helpers.batch_quotes(["159852", "159869"])
    assert res["159852"]["close"] == 0.663
    assert res["159869"]["close"] == 1.079
    assert res["159852"]["amount"] == 5000000


def test_batch_quotes_handles_bad_lines(monkeypatch):
    stdout = (
        "| symbol | date | open | last | high | low | volume | amount | exchange |\n"
        "| sh159852 | 2026-08-25 | 0.66 | 0.663 | 0.67 | 0.65 | 1000000 | 5000000 | 1.2 |\n"
        "| broken line |\n"
        "| sh159869 | 2026-08-25 | x | y | z | w | a | b | c |\n"
    )
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run_factory(stdout))
    res = westock_helpers.batch_quotes(["159852", "159869"])
    assert "159852" in res
    assert "159869" not in res  # 坏行被跳过而非崩溃
