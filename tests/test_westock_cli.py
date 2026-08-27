"""westock_cli 单元测试：代码前缀路由 + 表格解析 + 数值强转（正常/边界/异常）。

重点: _to_ws_code 现已**委托**给 westock_helpers._to_ws（单一事实来源, 见 B 修复），
沪市 ETF(5xxxx) 正确路由到 sh。此测试同时保证委托后行为等价、ETF 前缀不再漏修。
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
    # 沪市 ETF(5xxxx) 经委托正确路由到 sh（B 修复: 单一事实来源, 不再漏修）
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


def test_sector_mapping_normalizes_cached_int_keys(monkeypatch):
    """缓存经 data.to_json()->pd.read_json() 会把 '000001' 推断成 int 1,
    丢失前导零; 读回应规范回 6 位零填充字符串, 否则 afternoon_review 用
    zfill(6) 字符串 map 会全部 miss -> fillna('综合') 退化成单一大类。"""
    import pandas as pd
    from tushare_provider import TushareProvider

    # 模拟脏缓存: code 列被存成 int (1 / 300750 / 600519)
    bad = pd.DataFrame({
        "code": [1, 300750, 600519],
        "sector": ["银行", "电气设备", "白酒"],
    })

    class FakeDB:
        def cache_get(self, key):
            return bad

        def cache_put(self, *a, **k):
            pass

    # db 是只读 property -> get_db(); 通过 monkeypatch get_db 注入 FakeDB
    import tushare_provider
    monkeypatch.setattr(tushare_provider, "get_db", lambda: FakeDB())

    prov = TushareProvider()
    m = prov.get_sector_mapping()
    assert m == {"000001": "银行", "300750": "电气设备", "600519": "白酒"}, m
    assert all(isinstance(k, str) and len(k) == 6 and k.isdigit() for k in m)

