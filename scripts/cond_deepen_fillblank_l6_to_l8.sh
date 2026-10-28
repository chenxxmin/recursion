#!/bin/bash
# 二级条件监视器：randpos len3 l6 跑完 → 判定 → 仍有失败则启动 l8
cd /home/cxm/recursion
PYBIN=/home/cxm/miniconda3/bin/python
J6=experiments/variant_o2_fillblank_noaux_randpos_d512l6_P127_len3.json
J8=experiments/variant_o2_fillblank_noaux_randpos_d512l8_P127_len3.json
D6=/data/cxm/recursion/variant_o2_fillblank_noaux_randpos_d512l6_P127_len3/logs

echo "[$(date '+%F %T')] waiting for l6 randpos len3 batch ..."
while pgrep -f "batch_run.py .*$J6" > /dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] l6 batch done, judging:"

FAIL=0
for f in "$D6"/*.log; do
  n=$(basename "$f" .log | grep -oP 'a\d+b\d+')
  best=$(grep -oP 'Best=\K[\d.]+' "$f" | tail -1)
  conv=$(awk "BEGIN{print (${best:-0} >= 99.0) ? 1 : 0}")
  echo "  $n best=${best:-?} converged=$conv"
  [ "$conv" = "1" ] || FAIL=1
done

if [ "$FAIL" = "1" ]; then
  echo "[$(date '+%F %T')] failures present -> launching l8 deepened batch"
  BATCH_RUN_GPUS=0,1,3,4,5,6 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
    "$PYBIN" -u src/batch_run.py \
    --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
    "$J8" > "logs/run_variant_o2_fillblank_noaux_randpos_d512l8_P127_len3.log" 2>&1 < /dev/null &
  sleep 15
  pgrep -f "batch_run.py .*$J8" > /dev/null 2>&1 \
    && echo "[$(date '+%F %T')] LAUNCHED_L8" || { echo "[$(date '+%F %T')] L8_START_FAILED"; exit 1; }
else
  echo "[$(date '+%F %T')] ALL_CONVERGED_AT_L6 no l8 needed"
fi
