#!/bin/bash
# 0901 起 18h 编排：A线（N3 len1 fp32）GPU 2,3,6,7 / B线（d2048 len2 N2 fp32）GPU 0,1,4,5
# B线 d2048l2 已在 GPU 1,5 直接启动（本脚本不管）；
# 本脚本三个等待器：
#   1) l4r2 结束      -> GPU 2,3 起 A线 d1024l2 链（4 轮）
#   2) l6 首跑结束    -> GPU 6,7 先 l6r2 链（2 轮），再 d1024l4 链（5 轮）
#   3) d2048 r2 结束  -> GPU 0,4 起 B线 d2048l4 链（7 轮）
set -u
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }

(
  while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen1_n3_depth_fp32_l4r2\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] l4r2 结束，GPU 2,3 起 d1024l2 链..."
  bash scripts/chain_rounds.sh experiments/chain_a_d1024l2.json 2,3 4
) &

(
  while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen1_n3_depth_fp32\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] l6 首跑结束，GPU 6,7 起 l6r2 链..."
  bash scripts/chain_rounds.sh experiments/action_misslen1_n3_depth_fp32_l6r2.json 6,7 2
  echo "[$(ts)] l6r2 结束，GPU 6,7 起 d1024l4 链..."
  bash scripts/chain_rounds.sh experiments/chain_a_d1024l4.json 6,7 5
) &

(
  while pgrep -u "$(id -u)" -f 'batch_run\.py experiments/action_misslen2_n2_d2048_resume2\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] d2048 r2 结束，GPU 0,4 起 d2048l4 链..."
  bash scripts/chain_rounds.sh experiments/chain_b_d2048l4.json 0,4 7
) &

wait
echo "[$(ts)] 全部链结束"
