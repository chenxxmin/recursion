#!/bin/bash
# 等自己的训练进程结束后，运行：
#   experiments/addition_p127_tr64_ood128_misslen_l3plus.json
#   （missing 深层网格：N_LAYER 3..miss_len+2 x prob 0.1~0.3 x 8 种子，144 runs）
# 用法（在仓库根目录）：bash run_queued.sh
# 建议配合 tmux 或 nohup 使用：nohup bash run_queued.sh > queued_run.log 2>&1 &
#
# 说明：等待条件只匹配本用户的训练进程（batch_run.py / core.run_experiment），
# 其他用户的 python 不影响——batch_run.py 会自动挑选空闲 GPU
# （nvidia-smi 显存 <100MB 视为空闲），别人的进程占着的卡会被自动跳过。

set -u
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SEC=60
# 训练进程特征：batch_run 主进程 或 它派生的 core.run_experiment 子进程
TRAIN_PATTERN='batch_run\.py|from core import run_experiment'

ts() { date '+%F %T'; }

echo "[$(ts)] 等待自己的训练进程结束（每 ${POLL_SEC}s 检查一次）..."
while pgrep -u "$(id -u)" -f "$TRAIN_PATTERN" > /dev/null 2>&1; do
    echo "[$(ts)] 仍有 $(pgrep -u "$(id -u)" -fc "$TRAIN_PATTERN") 个训练进程在运行，继续等待..."
    sleep "$POLL_SEC"
done

echo "[$(ts)] 已无自己的训练进程。拉取最新代码..."
git pull || { echo "[$(ts)] git pull 失败，终止"; exit 1; }

echo "[$(ts)] 开始批次 1/1: addition_p127_tr64_ood128_misslen_l3plus.json"
"$PYTHON_BIN" src/batch_run.py experiments/addition_p127_tr64_ood128_misslen_l3plus.json
echo "[$(ts)] 批次 1/1 结束（batch_run 退出码 $?）"

echo "[$(ts)] 全部完成。"
