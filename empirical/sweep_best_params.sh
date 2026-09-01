#!/usr/bin/env bash
# 参数扫描: 在 v4.29 固化权重(vol0.5/rev0.5/q0) + lowvol_rev 真alpha 路线下,
# 扫描运营参数, 寻找样本内收益最高且回撤可控的配置。
# 注意: 权重本身不做网格搜索(方法论铁律: 权重须 OOS 验证, M2 已证 vol0.5/rev0.5/q0 为最高 OOS)。
# 2026-09-01: market.db 已切 DELETE 模式(无 WAL/SHM 锁抖动), 不再手动 rm 锁文件, 组间仅 sleep 释放句柄。
set -u
PY=D:/env/python3.12/python.exe
cd /d/workspace/github/a-stock-engine
OUT=data_cache/sweep_results.txt
: > "$OUT"
echo "sweep start $(date)" >> "$OUT"

run_one() {
  local name="$1"; shift
  echo "===== $name =====" >> "$OUT"
  "$PY" local_backtest.py "$@" > "data_cache/sweep_${name}.log" 2>&1
  echo "exit=$?" >> "$OUT"
  grep -E "总收益:|年化收益:|最大回撤:|夏普比率:" "data_cache/sweep_${name}.log" >> "$OUT"
  echo "" >> "$OUT"
  sleep 5   # 让 OS 释放文件句柄, 再启动下一组
}

# 基线(默认 lowvol_rev mh30 sl8 mps0.80 idle0): 已知 ~+12.94%, 重跑入表做对照
run_one base
# 策略对照: trend 应为反 alpha (预期 0 交易或负)
run_one trend   --alpha-mode trend
# 运营参数扫描
run_one mps085  --min-pick-score 0.85
run_one mps075  --min-pick-score 0.75
run_one mh45    --min-hold 45
run_one sl6     --stop-loss 6
run_one idle    --idle-cash-rate 0.02

echo "sweep done $(date)" >> "$OUT"
