#!/bin/bash
# 等线B (action_misslen2_n2_scaleup) 的 batch_run 结束后，
# 在 GPU 6-7 上跑 A2b（N3 d1024l4 大实验，与 B 续跑的 4-5 互不冲突）。
set -u
cd "$(dirname "$0")/.."

ts() { date '+%F %T'; }

echo "[$(ts)] 等待线B batch_run 结束..."
while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen2_n2_scaleup\.json' > /dev/null 2>&1; do
    sleep 60
done

echo "[$(ts)] 线B已结束，启动 A2b（N3 d1024l4，GPU 6-7）..."
BATCH_RUN_GPUS=6,7 python -u src/batch_run.py experiments/action_misslen1_n3_scaleup_big.json
echo "[$(ts)] A2b 批次退出（码 $?）"
