"""pick_tracker 单元测试：命中追踪写入/查询（用临时 DB 隔离）。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import pick_tracker as pt


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "TRACKING_DB", str(tmp_path / "tracking.db"))
    return pt.PickTracker()


def test_track_picks_records(tracker):
    df = pd.DataFrame({
        "code": ["600519", "000001"],
        "name": ["茅台", "平安"],
    })
    n = tracker.track_picks(df, "pre_market", "2026-08-26", "②A_质量榜")
    assert n == 2


def test_track_picks_empty(tracker):
    assert tracker.track_picks(pd.DataFrame(), "pre_market") == 0


def test_track_picks_dedup_same_day(tracker):
    df = pd.DataFrame({"code": ["600519"], "name": ["茅台"]})
    n1 = tracker.track_picks(df, "pre_market", "2026-08-26", "②A_质量榜")
    n2 = tracker.track_picks(df, "pre_market", "2026-08-26", "②A_质量榜")
    # 同日同 code 去重: 第二次不新增
    assert n1 == 1
    assert n2 == 0


def test_get_summary_after_track(tracker):
    df = pd.DataFrame({"code": ["600519"], "name": ["茅台"]})
    tracker.track_picks(df, "pre_market", "2026-08-26", "②A_质量榜")
    summary = tracker.get_summary("pre_market")
    assert summary is not None
