#!/bin/bash
# len5 randpos noaux 尺寸对照：先 d1024l4，跑完接 d512l6
cd /home/cxm/recursion
PYBIN=/home/cxm/miniconda3/bin/python
J1=experiments/variant_o2_fillblank_noaux_randpos_d1024l4_P127_len5.json
J2=experiments/variant_o2_fillblank_noaux_randpos_d512l6_P127_len5.json
run_batch () {
  local J=$1; local B=$(basename "$J" .json)
  echo "[$(date '+%F %T')] starting $B"
  BATCH_RUN_GPUS=0,1,3,4,5,6 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
    "$PYBIN" -u src/batch_run.py \
    --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
    "$J" > "logs/run_${B}.log" 2>&1 < /dev/null &
  sleep 15
  pgrep -f "batch_run.py .*$J" > /dev/null 2>&1 \
    && echo "[$(date '+%F %T')] LAUNCHED $B" || { echo "[$(date '+%F %T')] START_FAILED $B"; exit 1; }
  while pgrep -f "batch_run.py .*$J" > /dev/null 2>&1; do sleep 60; done
  echo "[$(date '+%F %T')] DONE $B"
}
run_batch "$J1"
run_batch "$J2"
echo "[$(date '+%F %T')] BOTH_DONE"
