#!/bin/bash
# fill_blank noaux 五组实验链式监控：前一批 batch_run 退出后启动下一批。
# 每分钟检查一次；只匹配本链自己的 JSON 路径，不会误认 zkh 的批。
cd /home/cxm/recursion
PYBIN=/home/cxm/miniconda3/bin/python
GPUS=0,1,3,4,5,6

QUEUE=(
  experiments/variant_o2_fillblank_noaux_fixpos1_d512l4_P127_len3.json
  experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3.json
  experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len5.json
  experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len8.json
)
PREV=experiments/variant_o2_fillblank_noaux_fixpos0_d512l4_P127_len3.json

for JSON in "${QUEUE[@]}"; do
  B=$(basename "$JSON" .json)
  echo "[$(date '+%F %T')] waiting for $PREV to finish ..."
  while pgrep -f "batch_run.py .*$PREV" > /dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date '+%F %T')] $PREV done; starting $B on GPUs $GPUS"
  BATCH_RUN_GPUS=$GPUS BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
    "$PYBIN" -u src/batch_run.py \
    --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
    "$JSON" > "logs/run_${B}.log" 2>&1 < /dev/null &
  sleep 10
  if ! pgrep -f "batch_run.py .*$JSON" > /dev/null 2>&1; then
    echo "[$(date '+%F %T')] ERROR: $B failed to start, check logs/run_${B}.log; chain stopped"
    exit 1
  fi
  PREV=$JSON
done
echo "[$(date '+%F %T')] all 5 batches launched; chain monitor exiting"
