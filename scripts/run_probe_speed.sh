#!/bin/bash
# 等线B batch 结束（GPU 6/7 随即可用）后，在 GPU 6 上跑加速探针
set -u
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }

echo "[$(ts)] 等待线B batch_run 结束..."
while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen2_n2_scaleup\.json' > /dev/null 2>&1; do
    sleep 60
done
echo "[$(ts)] 线B已结束，在 GPU 6 上启动探针..."
CUDA_VISIBLE_DEVICES=6 python -u scripts/probe_speed.py
echo "[$(ts)] 探针结束（码 $?）"
