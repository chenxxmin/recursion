#!/bin/bash
# 等服务器上现有 python 进程全部结束后，依次运行：
#   1. experiments/dynamic_mixed_large.json
#   2. experiments/nonlinear.json
# 用法（在仓库根目录）：bash run_queued_experiments.sh
# 建议配合 tmux 或 nohup 使用：nohup bash run_queued_experiments.sh > queued_run.log 2>&1 &

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

echo "[$(ts)] 开始批次 1/2: dynamic_mixed_large.json"
"$PYTHON_BIN" src/batch_run.py experiments/dynamic_mixed_large.json
echo "[$(ts)] 批次 1/2 结束（batch_run 退出码 $?）"

echo "[$(ts)] 开始批次 2/2: nonlinear.json"
"$PYTHON_BIN" src/batch_run.py experiments/nonlinear.json
echo "[$(ts)] 批次 2/2 结束（batch_run 退出码 $?）"

echo "[$(ts)] 全部完成。"
