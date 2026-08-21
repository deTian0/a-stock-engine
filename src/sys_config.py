"""
sys_config.py — 系统级配置单例（编码/npx路径等）
"""

import os
from pathlib import Path
import yaml


_CONFIG = None


def _load():
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f)
        _CONFIG = full.get("system", {})
    except Exception:
        _CONFIG = {}
    return _CONFIG


def get_encoding() -> str:
    """获取系统编码，默认 utf-8。"""
    cfg = _load()
    enc = cfg.get("encoding", "utf-8")
    if enc in ("auto", "default", ""):
        enc = "utf-8"
    return enc


def get_npx_path() -> str:
    """获取 npx 路径。"""
    cfg = _load()
    path = cfg.get("npx_path", "npx.cmd")
    # 支持环境变量 $HOME
    path = os.path.expandvars(path)
    if os.path.exists(path):
        return path
    # 自动找 WorkBuddy node
    home = Path.home()
    for ver in ["22.22.2", "20.0.0"]:
        candidate = home / ".workbuddy" / "binaries" / "node" / "versions" / ver / "npx.cmd"
        if candidate.exists():
            return str(candidate)
    return "npx.cmd"
