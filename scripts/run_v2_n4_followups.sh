#!/bin/bash
# 0903 下午编排：
#   1) l1 收尾链（GPU 1,3）结束后 -> GPU 1,3 起 N4×0.1 链
#   2) N2×0.3（GPU 0,4）和 N3×0.1（GPU 5,6）都结束后 -> GPU 0,4,5,6 起 N4×0.3 链
set -u
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }
export CHAIN_DATA_BASE=/data/cxm/recursion_v2 CHAIN_MODEL_BASE=/data/cxm/models_v2

(
  while pgrep -u "$(id -u)" -f 'chain_rounds\.sh experiments/v2/action_v2_d1024_n2_miss01\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] l1 收尾结束，GPU 1,3 起 N4 miss0.1 链..."
  bash scripts/chain_rounds.sh experiments/v2/action_v2_d1024_n4_miss01.json 1,3 30
) &

(
  while pgrep -u "$(id -u)" -f 'chain_rounds\.sh experiments/v2/action_v2_d1024_n2_miss03\.json' > /dev/null 2>&1 \
     || pgrep -u "$(id -u)" -f 'chain_rounds\.sh experiments/v2/action_v2_d1024_n3_miss01\.json' > /dev/null 2>&1; do sleep 60; done
  echo "[$(ts)] N2x0.3 与 N3x0.1 均结束，GPU 0,4,5,6 起 N4 miss0.3 链..."
  bash scripts/chain_rounds.sh experiments/v2/action_v2_d1024_n4_miss03.json 0,4,5,6 30
) &

wait
echo "[$(ts)] 全部编排结束"
