#!/bin/bash
cd /home/cxm/recursion
J1=experiments/variant_o2_corruptfix_valueonly_d512l4_P127_len5_pos04.json
J2=experiments/variant_o2_corruptfix_valueonly_d512l4_P127_len5_initPos04.json
run_batch () {
  local J=$1
  local B
  B=$(basename "$J" .json)
  echo "[$(date '+%F %T')] starting $B"
  BATCH_RUN_GPUS=0,3,4,6,7 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
    /home/cxm/miniconda3/bin/python -u src/batch_run.py \
    --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
    "$J" > "logs/run_${B}.log" 2>&1 < /dev/null &
  sleep 15
  pgrep -f "batch_run.py .*$J" > /dev/null 2>&1 || { echo "START_FAILED $B"; exit 1; }
  while pgrep -f "batch_run.py .*$J" > /dev/null 2>&1; do sleep 60; done
  echo "[$(date '+%F %T')] DONE $B"
}
run_batch "$J1"
run_batch "$J2"
echo "[$(date '+%F %T')] BOTH_DONE"
