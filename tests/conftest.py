"""a-stock-engine 测试公共夹具与假数据构造器。

说明: 本项目为 Python 工程，按等价方式使用 pytest（Java 的 JUnit 在 Python 不适用）。
所有网络/子进程调用均通过 monkeypatch 替换为内存假数据，保证测试确定性、可离线运行。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def config():
    """加载仓库真实 config.yaml（含已对齐华宝实盘的 3 只 ETF 持仓）。"""
    return yaml.safe_load(open(REPO_ROOT / "config.yaml", encoding="utf-8"))


@pytest.fixture
def regime_bull():
    return {
        "regime": "多头",
        "position_cap": 0.80,
        "judgment": "测试-多头",
        "indices": {
            "000001": {"name": "上证指数", "close": 3500, "ma_short": 3480,
                       "ma_long": 3400, "above_ma": True},
        },
    }


@pytest.fixture
def regime_bear():
    return {
        "regime": "空头",
        "position_cap": 0.20,
        "judgment": "测试-空头",
        "indices": {},
    }


@pytest.fixture
def base_results(regime_bull):
    """generate_brief 的最小可运行 inputs。"""
    return {
        "timestamp": "2026-08-25 23:00",
        "elapsed_seconds": 1.0,
        "regime": regime_bull,
        "categories": {},
        "etf_picks": pd.DataFrame(),
        "l4_results": pd.DataFrame(),
        "rebound_picks": [],
        "l2_filtered_count": 0,
    }


@pytest.fixture
def sample_lvrev_df():
    """同时覆盖 score_lvrev 与 apply_entry_gates 所需列。"""
    return pd.DataFrame({
        "code": ["600519", "000001", "300750"],
        "name": ["茅台", "平安", "宁德"],
        "vol20": [0.1, 0.3, 0.5],
        "rev_chg": [-0.05, -0.02, 0.03],
        "debt_ratio": [30, 60, 45],
        "revenue_growth": [10, -5, 20],
        "pb": [10, 0.8, 5],
        "ps_ttm": [5, 1, 3],
        "pe": [30, 8, 50],
        "close": [1800, 15, 200],
        "ma20": [1800, 15, 195],
        "ma60": [1700, 14, 190],
        "rs20": [0.5, 0.3, 0.6],
    })


# ---------------------------------------------------------------------------
# 假子进程输出构造器（替代 westock-data CLI 的 subprocess.run）
# ---------------------------------------------------------------------------
class FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def make_kline_stdout(ws_code: str, closes: list[float]) -> str:
    """生成 westock kline 命令的假 stdout。

    每行格式: | symbol | date | open | close | high | low | volume | amount | ex |
    列顺序与 westock_helpers.batch_kline / batch_quotes 的解析一致(parts[4]=close/last)。
    """
    lines = ["| symbol | date | open | close | high | low | volume | amount | ex |"]
    for i, c in enumerate(closes):
        lines.append(
            f"| {ws_code} | 2026-08-{25 - i:02d} | {c - 0.01:.2f} | {c:.2f} | "
            f"{c + 0.01:.2f} | {c - 0.02:.2f} | 1000 | 100000 | 1.0 |"
        )
    return "\n".join(lines)


def fake_subprocess_run_factory(kline_stdout: str):
    """返回一个可替 subprocess.run 的工厂：任何 kline 调用都返回同一假 stdout。"""
    def _run(args, **kwargs):
        # batch_quotes / batch_kline 都以 "kline" 子命令调用 westock-data
        if isinstance(args, (list, tuple)) and "kline" in args:
            return FakeCompletedProcess(kline_stdout)
        return FakeCompletedProcess("")
    return _run
