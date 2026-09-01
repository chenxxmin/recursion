#!/bin/bash
# 等 A2b 和 B续跑 两个 batch 都结束后，在 GPU 3,4,5,6,7 上跑 N3 深度扫描
set -u
cd "$(dirname "$0")/.."

ts() { date '+%F %T'; }

echo "[$(ts)] 等待 A2b 和 B-resume 批次结束..."
while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen1_n3_scaleup_big\.json' > /dev/null 2>&1 \
   || pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen2_n2_d1024l4_resume\.json' > /dev/null 2>&1; do
    sleep 60
done

echo "[$(ts)] 两个批次都已结束，启动 N3 深度扫描（GPU 3,4,5,6,7）..."
BATCH_RUN_GPUS=3,4,5,6,7 python -u src/batch_run.py experiments/action_misslen1_n3_depth.json
echo "[$(ts)] 深度扫描批次退出（码 $?）"
