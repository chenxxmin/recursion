#!/bin/bash
# 等 N3 深度扫描结束后，在空闲卡上跑 fp32 vs bf16 对照实验
set -u
cd "$(dirname "$0")/.."

ts() { date '+%F %T'; }

echo "[$(ts)] 等待深度扫描批次结束..."
while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen1_n3_depth\.json' > /dev/null 2>&1; do
    sleep 60
done

echo "[$(ts)] 深度扫描已结束，启动 fp32 vs bf16 对照（GPU 4,5,6,7 中的空闲卡）..."
BATCH_RUN_GPUS=4,5,6,7 python -u src/batch_run.py experiments/action_fp32_vs_bf16_d256l2n3.json
echo "[$(ts)] 对照实验批次退出（码 $?）"
