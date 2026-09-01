#!/bin/bash
# 编排：l4 fp32 续跑已在 GPU 0,1 直接启动（本脚本不管）；
# 本脚本负责两个等待器：
#   1) depth_fp32 批次（l6）结束后 -> GPU 6,7 续跑 l6
#   2) d2048 resume2 批次结束后 -> GPU 2,3,4,5 续跑 d2048 round3
set -u
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }

(
  while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen1_n3_depth_fp32\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] depth_fp32 批次结束，GPU 6,7 续跑 l6..."
  BATCH_RUN_GPUS=6,7 python -u src/batch_run.py experiments/action_misslen1_n3_depth_fp32_l6r2.json
  echo "[$(ts)] l6 续跑退出（码 $?）"
) &

(
  while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen2_n2_d2048_resume2\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] d2048 resume2 批次结束，GPU 0,4,5 续跑 round3..."
  BATCH_RUN_GPUS=0,4,5 python -u src/batch_run.py experiments/action_misslen2_n2_d2048_resume3.json
  echo "[$(ts)] d2048 round3 退出（码 $?）"
) &

wait
echo "[$(ts)] 全部后续批次结束"
