#!/bin/bash
# 等 len5 randpos l4 批跑完后，用全部非 zkh 卡（0,1,3,4,5,6,7）启动 len3to5 不定长批
cd /home/cxm/recursion
PYBIN=/home/cxm/miniconda3/bin/python
J5=experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len5.json
J35=experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5.json

echo "[$(date '+%F %T')] waiting for len5 batch ..."
while pgrep -f "batch_run.py .*$J5" > /dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] len5 done; starting len3to5 on GPUs 0,1,3,4,5,6,7"
BATCH_RUN_GPUS=0,1,3,4,5,6,7 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
  "$PYBIN" -u src/batch_run.py \
  --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
  "$J35" > "logs/run_variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5.log" 2>&1 < /dev/null &
sleep 15
pgrep -f "batch_run.py .*$J35" > /dev/null 2>&1 \
  && echo "[$(date '+%F %T')] LAUNCHED_LEN3TO5" || { echo "[$(date '+%F %T')] START_FAILED"; exit 1; }
