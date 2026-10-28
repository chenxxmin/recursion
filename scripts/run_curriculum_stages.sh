#!/bin/bash
cd /home/cxm/recursion
PYBIN=/home/cxm/miniconda3/bin/python
for J in $(python3 -c "import json;print('\n'.join(json.load(open('scripts/curriculum_stages.json'))))"); do
  B=$(basename "$J" .json)
  echo "[$(date '+%F %T')] starting $B"
  BATCH_RUN_GPUS=0,3,4,6,7 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
    "$PYBIN" -u src/batch_run.py --base-dir /data/cxm/recursion \
    --model-base-dir /data/cxm/models "$J" > "logs/run_${B}.log" 2>&1 < /dev/null &
  sleep 15
  pgrep -f "batch_run.py .*$J" > /dev/null 2>&1 || { echo "START_FAILED $B"; exit 1; }
  while pgrep -f "batch_run.py .*$J" > /dev/null 2>&1; do sleep 60; done
  echo "[$(date '+%F %T')] DONE $B"
done
echo "[$(date '+%F %T')] ALL_STAGES_DONE"
