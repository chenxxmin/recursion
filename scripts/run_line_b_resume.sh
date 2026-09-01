#!/bin/bash
# 等线B (action_misslen2_n2_scaleup) 的 batch_run 结束后，
# 在 GPU 4-7 上续跑 d1024l4 miss0.1 的两个超时实验。
set -u
cd "$(dirname "$0")/.."

ts() { date '+%F %T'; }

echo "[$(ts)] 等待线B batch_run 结束..."
while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen2_n2_scaleup\.json' > /dev/null 2>&1; do
    sleep 60
done

echo "[$(ts)] 线B已结束，启动 d1024l4 miss0.1 续跑（GPU 4-7）..."
BATCH_RUN_GPUS=4,5,6,7 python -u src/batch_run.py experiments/action_misslen2_n2_d1024l4_resume.json
echo "[$(ts)] 续跑批次退出（码 $?）"
