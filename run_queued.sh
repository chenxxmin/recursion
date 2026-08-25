#!/bin/bash
# 等自己的训练进程结束后，运行：
#   experiments/mixed_basic_d256l1r8h2_p127_rules12345.json
#   （mixed basic 无 tag：N=2..5 规则 [(1,1),(2,3),(3,5),(4,7),(5,11)] x 4 seeds，16 runs，7 并发）
# 用法（在仓库根目录）：bash run_queued.sh
# 建议配合 tmux 或 nohup 使用：nohup bash run_queued.sh > queued_run.log 2>&1 &
#
# 说明：等待条件只匹配本用户的训练进程（batch_run.py 主进程 / 它派生的 core.py 子进程），
# 其他用户的 python 不影响——batch_run.py 会自动挑选空闲 GPU
# （nvidia-smi 显存 <100MB 视为空闲），别人的进程占着的卡会被自动跳过。

set -u
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SEC=60
# 训练进程特征：batch_run 主进程 或 它派生的 core.py 训练子进程
TRAIN_PATTERN='batch_run\.py|core\.py'

ts() { date '+%F %T'; }

echo "[$(ts)] 等待自己的训练进程结束（每 ${POLL_SEC}s 检查一次）..."
while pgrep -u "$(id -u)" -f "$TRAIN_PATTERN" > /dev/null 2>&1; do
    echo "[$(ts)] 仍有 $(pgrep -u "$(id -u)" -fc "$TRAIN_PATTERN") 个训练进程在运行，继续等待..."
    sleep "$POLL_SEC"
done

echo "[$(ts)] 已无自己的训练进程。拉取最新代码..."
git pull || { echo "[$(ts)] git pull 失败，终止"; exit 1; }

echo "[$(ts)] 开始: mixed_basic_d256l1r8h2_p127_rules12345.json"
"$PYTHON_BIN" src/batch_run.py experiments/mixed_basic_d256l1r8h2_p127_rules12345.json
echo "[$(ts)] 结束（batch_run 退出码 $?）"

echo "[$(ts)] 全部完成。"
