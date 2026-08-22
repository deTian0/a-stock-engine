# -*- coding: utf-8 -*-
"""v4.25 隔离 ablation: 单独开关 lvrev_scorer 的 A / B 子改动, 定位回退真因。
- A: 仅 W_DEFAULT 重加权 (vol/rev↑, q↓), 保留 ROE 正向排序
- B: 仅去 ROE 正向排序 (s_q 仅 debt_ratio), 保留原 W_DEFAULT
注: 补丁的 C/D (multifactor 估值/质量闸门) 对回测零影响(回测走 lvrev_scorer.score_lvrev,
     不调用 multifactor._score_l4_lvrev), 故此处不参与 ablation。
每次跑完用 git checkout 还原 src/lvrev_scorer.py, 保证工作树干净。
"""
import subprocess, os, sys

REPO = r"D:\workspace\github/a-stock-engine"
SRC = os.path.join(REPO, "src", "lvrev_scorer.py")
INTERP = r"D:\env\python3.12\python.exe"

W_DEFAULT_OLD = 'W_DEFAULT = dict(vol=0.45, rev=0.35, value=0.0, q=0.12, g=0.08)'
W_DEFAULT_NEW = 'W_DEFAULT = dict(vol=0.52, rev=0.43, value=0.0, q=0.05, g=0.08)  # ABL-A'
W_VALUE_OLD = 'W_VALUE = dict(vol=0.38, rev=0.27, value=0.18, q=0.10, g=0.07)'
W_VALUE_NEW = 'W_VALUE = dict(vol=0.43, rev=0.32, value=0.18, q=0.05, g=0.07)  # ABL-A'

ROE_BLOCK_OLD = (
    '    # 3) 质量: ROE 越高越好, 负债率越低越好\n'
    '    s_q = pd.Series(0.5, index=d.index)\n'
    '    if "roe" in d.columns:\n'
    '        s_q = s_q + d["roe"].clip(-20, 40).fillna(0) / 40.0'
)
ROE_BLOCK_NEW = (
    '    # 3) 质量稳定器: 仅保留低杠杆(debt_ratio), 不再用 ROE 正向排序 (ABL-B)\n'
    '    s_q = pd.Series(0.5, index=d.index)'
)

# 每个 variant = 一组 (old, new) 替换, 要求各自恰好命中 1 次
VARIANTS = {
    "A": [(W_DEFAULT_OLD, W_DEFAULT_NEW), (W_VALUE_OLD, W_VALUE_NEW)],
    "B": [(ROE_BLOCK_OLD, ROE_BLOCK_NEW)],
}


def apply_and_run(variant, log_path):
    with open(SRC, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new in VARIANTS[variant]:
        cnt = src.count(old)
        if cnt != 1:
            raise SystemExit(f"[ABL-{variant}] 替换未命中1次 (实际{cnt}): {old[:40]!r}")
        src = src.replace(old, new)
    with open(SRC, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[ABL-{variant}] 已应用, 启动回测 -> {log_path}", flush=True)
    rc = subprocess.run(
        [INTERP, "local_backtest.py"], cwd=REPO,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    ).returncode
    print(f"[ABL-{variant}] 回测结束 rc={rc}", flush=True)
    # 还原
    subprocess.run(["git", "checkout", "--", "src/lvrev_scorer.py"], cwd=REPO)
    print(f"[ABL-{variant}] 已 git checkout 还原", flush=True)


if __name__ == "__main__":
    order = sys.argv[1:] or ["A", "B"]
    for v in order:
        if v not in VARIANTS:
            raise SystemExit(f"未知 variant: {v}")
        apply_and_run(v, os.path.join(REPO, f"empirical/logs/bt_ablation_{v}.log"))
    print("[ABL] 全部 ablation 完成", flush=True)
