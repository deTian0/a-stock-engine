"""westock_cli 单元测试：代码前缀路由 + 表格解析 + 数值强转（正常/边界/异常）。

重点: _to_ws_code 与已修的 westock_helpers._to_ws 有**同样的 ETF 前缀 bug**——
沪市 ETF(5xxxx) 会被错路由成 sz，这是主 CLI 路径的真实 bug。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import westock_cli


def test_to_ws_code_sh():
    assert westock_cli._to_ws_code("600519") == "sh600519"


def test_to_ws_code_sz():
    assert westock_cli._to_ws_code("000001") == "sz000001"
    assert westock_cli._to_ws_code("159852") == "sz159852"   # 深市 ETF


def test_to_ws_code_bj():
    assert westock_cli._to_ws_code("889999") == "bj889999"


def test_to_ws_code_etf_routing_shanghai():
    # BUG 捕获: 沪市 ETF(5xxxx) 必须路由到 sh, 当前返回 sz (错误)
    assert westock_cli._to_ws_code("515790") == "sh515790"   # 光伏ETF
    assert westock_cli._to_ws_code("510050") == "sh510050"   # 上证50ETF
    assert westock_cli._to_ws_code("512880") == "sh512880"   # 证券ETF


def test_to_index_ws_code_shanghai_composite():
    # 上证指数 000001 撞码平安银行, 必须走 sh
    assert westock_cli._to_index_ws_code("000001") == "sh000001"


def test_to_index_ws_code_sz_index():
    assert westock_cli._to_index_ws_code("399001") == "sz399001"
    assert westock_cli._to_index_ws_code("399006") == "sz399006"


def test_parse_pipe_table_normal():
    out = "| code | name | close |\n|------|------|------|\n| 600519 | 茅台 | 1800 |\n| 000001 | 平安 | 15 |"
    rows = westock_cli._parse_pipe_table(out)
    assert len(rows) == 2
    assert rows[0]["code"] == "600519"
    assert rows[1]["close"] == "15"


def test_parse_pipe_table_skips_separator():
    out = "| code | name |\n|------|------|\n| 600519 | 茅台 |"
    rows = westock_cli._parse_pipe_table(out)
    assert len(rows) == 1
    assert rows[0]["name"] == "茅台"


def test_parse_pipe_table_too_short_returns_empty():
    assert westock_cli._parse_pipe_table("| code | name |") == []
    assert westock_cli._parse_pipe_table("") == []


def test_coerce_numeric_all_numeric():
    df = pd.DataFrame({"close": ["1.0", "2.0", "3.0"], "name": ["a", "b", "c"]})
    out = westock_cli._coerce_numeric(df)
    assert pd.api.types.is_numeric_dtype(out["close"])
    assert out["close"].tolist() == [1.0, 2.0, 3.0]
    # name 含非数字, 不应被强转(pandas 3.x 下为 StringDtype, 非 object)
    assert not pd.api.types.is_numeric_dtype(out["name"])


def test_coerce_numeric_mixed_text_keeps_original():
    # 含无法解析的文本 -> 该列保留原样(不破坏)
    df = pd.DataFrame({"x": ["1.0", "abc", "3.0"]})
    out = westock_cli._coerce_numeric(df)
    assert not pd.api.types.is_numeric_dtype(out["x"])


def test_sector_of_with_mapping_no_network():
    mapping = {"600519": "白酒", "000001": "银行"}
    assert westock_cli.sector_of("600519", mapping) == "白酒"
    assert westock_cli.sector_of("999999", mapping) == "未知"
