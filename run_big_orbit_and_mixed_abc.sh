#!/bin/bash
# 等服务器上现有 python 进程全部结束后，依次运行：
#   1. experiments/addition_big_orbit_p53.json   （大轨道 addition：a2b2/a1b5/a3b3，36 runs）
#   2. experiments/mixed_abc_basic.json          （mixed_abc scale-up + P31，36 runs）
# 用法（在仓库根目录）：bash run_big_orbit_and_mixed_abc.sh
# 建议配合 tmux 或 nohup 使用：nohup bash run_big_orbit_and_mixed_abc.sh > queued_run.log 2>&1 &

set -u
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SEC=60

ts() { date '+%F %T'; }

echo "[$(ts)] 等待现有 python 进程结束（每 ${POLL_SEC}s 检查一次）..."
while pgrep -f python > /dev/null 2>&1; do
    echo "[$(ts)] 仍有 $(pgrep -fc python) 个 python 进程在运行，继续等待..."
    sleep "$POLL_SEC"
done

echo "[$(ts)] 已无 python 进程。拉取最新代码..."
git pull || { echo "[$(ts)] git pull 失败，终止"; exit 1; }

echo "[$(ts)] 开始批次 1/2: addition_big_orbit_p53.json"
"$PYTHON_BIN" src/batch_run.py experiments/addition_big_orbit_p53.json
echo "[$(ts)] 批次 1/2 结束（batch_run 退出码 $?）"

echo "[$(ts)] 开始批次 2/2: mixed_abc_basic.json"
"$PYTHON_BIN" src/batch_run.py experiments/mixed_abc_basic.json
echo "[$(ts)] 批次 2/2 结束（batch_run 退出码 $?）"

echo "[$(ts)] 全部完成。"
