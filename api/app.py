"""
A股多因子选股系统 — Flask 后端
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent          # api/
PROJECT_DIR = BACKEND_DIR.parent             # a-stock-engine/
sys.path.insert(0, str(PROJECT_DIR))          # 让 import api.xxx 工作
sys.path.insert(0, str(PROJECT_DIR / "src"))  # 复用引擎模块

from flask import Flask
from flask_cors import CORS

from api.select_api import select_bp
from api.backtest_api import backtest_bp
from api.data_api import data_bp


def create_app():
    app = Flask(__name__, static_folder=str(PROJECT_DIR / "front" / "dist"))
    CORS(app)

    app.register_blueprint(select_bp, url_prefix="/api/select")
    app.register_blueprint(backtest_bp, url_prefix="/api/backtest")
    app.register_blueprint(data_bp, url_prefix="/api/data")

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        static_dir = PROJECT_DIR / "front" / "dist"
        if path and (static_dir / path).exists():
            from flask import send_from_directory
            return send_from_directory(str(static_dir), path)
        from flask import send_from_directory
        return send_from_directory(str(static_dir), "index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
