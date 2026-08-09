"""
回测 API — 异步执行 + 结果查询
"""
import threading
import json
import uuid
import sys
from pathlib import Path
from datetime import datetime

from flask import Blueprint, jsonify, request

backtest_bp = Blueprint("backtest", __name__)

# 简单内存存储（生产环境应换 Redis/DB）
_backtest_jobs = {}


def _load_config():
    """加载回测配置。"""
    config_path = Path(__file__).parent.parent.parent / "config.selector.yaml"
    if not config_path.exists():
        return {}
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_backtest_worker(job_id: str, params: dict):
    """后台执行回测（在独立线程中）。"""
    try:
        config = _load_config()
        cap = int(params.get("capital",
                            config.get("portfolio", {}).get("initial_capital", 50000)))
        max_picks = int(params.get("max_picks",
                                  config.get("portfolio", {}).get("max_picks_per_day", 20)))

        # 导入回测模块
        engine_dir = Path(__file__).parent.parent.parent  # a-stock-engine/
        sys.path.insert(0, str(engine_dir))
        from local_backtest import LocalBacktest

        bt = LocalBacktest()
        pf = bt.run_portfolio()

        sell_stats = pf.get("sell_stats", {})
        _backtest_jobs[job_id] = {
            "status": "done",
            "result": {
                "initial": pf["initial"],
                "final": pf["final"],
                "return_pct": pf["return_pct"],
                "cagr_pct": pf["cagr_pct"],
                "max_drawdown_pct": pf["max_drawdown_pct"],
                "sharpe": pf["sharpe"],
                "years": pf["years"],
                "year_returns": pf.get("year_returns", {}),
                "portfolio": pf.get("portfolio", []),
                "trades": sell_stats.get("total_trades", 0),
                "avg_held": sell_stats.get("avg_held_days", 0),
                "avg_return": sell_stats.get("avg_return", 0),
                "sell_reasons": sell_stats.get("reasons", {}),
            },
        }
    except Exception as e:
        _backtest_jobs[job_id] = {"status": "error", "error": str(e)}


@backtest_bp.route("/start", methods=["POST"])
def start_backtest():
    """启动回测（异步）。"""
    body = request.get_json(silent=True) or {}
    job_id = str(uuid.uuid4())[:8]

    _backtest_jobs[job_id] = {"status": "running", "started_at": datetime.now().isoformat()}
    t = threading.Thread(target=_run_backtest_worker, args=(job_id, body), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "running"})


@backtest_bp.route("/status/<job_id>")
def backtest_status(job_id: str):
    """查询回测进度/结果。"""
    job = _backtest_jobs.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(job)


@backtest_bp.route("/config", methods=["GET", "POST"])
def backtest_config():
    """读取/保存回测配置。"""
    config_path = Path(__file__).parent.parent.parent / "config.selector.yaml"

    if request.method == "POST":
        new_config = request.get_json(silent=True) or {}
        import yaml
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False)
        return jsonify({"status": "saved"})

    config = _load_config()
    return jsonify(config)
