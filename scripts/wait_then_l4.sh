#!/bin/bash
# 等当前 l2 批次(batch_run PID $1)结束后，在 GPU4-7 启动 l4 批次
cd /mnt/workspace/hujiachen/recursion
while kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 10
BATCH_RUN_GPUS=4,5,6,7 /usr/bin/python3 -u src/batch_run.py experiments/tetranacci_p127_d1024l4_300k_rand4.json \
  > logs/run_q4-7_tetranacci_l4_rand4.log 2>&1
