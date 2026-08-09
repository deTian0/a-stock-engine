"""
选股 API
"""
from pathlib import Path
from flask import Blueprint, jsonify, request

from services.selector_service import SelectorService

select_bp = Blueprint("select", __name__)


@select_bp.route("/run", methods=["POST"])
def run_selection():
    """执行选股。"""
    body = request.get_json(silent=True) or {}
    top_n = int(body.get("top_n", 20))
    max_per_sector = int(body.get("max_per_sector", 5))

    svc = SelectorService()
    try:
        results = svc.run(top_n=top_n, max_per_sector=max_per_sector)
        return jsonify(results)
    finally:
        svc.close()


@select_bp.route("/latest")
def latest_picks():
    """获取最近一次选股结果。"""
    output_dir = Path(__file__).parent.parent.parent / "output"
    files = sorted(output_dir.glob("picks_*.json"), reverse=True)
    if not files:
        return jsonify({"error": "无历史选股记录", "picks": []})

    import json
    with open(files[0], encoding="utf-8") as f:
        return jsonify(json.load(f))
