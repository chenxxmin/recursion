#!/bin/bash
# 等 len3~5 补齐 ckpt 的批跑完，然后启动 len8 暖启动微调（6 系数）
cd /home/cxm/recursion
J1=experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5_wd2_ckpt_rest.json
J2=experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len8_wd2_initLen35.json
while pgrep -f "batch_run.py .*$J1" > /dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] ckpt repro done; launching len8 warm-start"
BATCH_RUN_GPUS=0,1,3,4,5,6 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
  /home/cxm/miniconda3/bin/python -u src/batch_run.py \
  --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
  "$J2" > logs/run_variant_o2_fillblank_noaux_randpos_d512l4_P127_len8_wd2_initLen35.log 2>&1 < /dev/null &
sleep 15
pgrep -f "batch_run.py .*$J2" > /dev/null 2>&1 && echo LAUNCHED || echo START_FAILED
