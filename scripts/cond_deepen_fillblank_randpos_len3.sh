#!/bin/bash
# 条件链：randpos len3 l4 跑完 → 判定 → 失败则启动 l6，并挂出 l6→l8 二级监视器
cd /home/cxm/recursion
PYBIN=/home/cxm/miniconda3/bin/python
J4=experiments/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3.json
J6=experiments/variant_o2_fillblank_noaux_randpos_d512l6_P127_len3.json
D4=/data/cxm/recursion/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3/logs

echo "[$(date '+%F %T')] waiting for l4 randpos len3 batch ..."
while pgrep -f "batch_run.py .*$J4" > /dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] l4 batch done, judging:"

FAIL=0
for f in "$D4"/*.log; do
  n=$(basename "$f" .log | grep -oP 'a\d+b\d+')
  best=$(grep -oP 'Best=\K[\d.]+' "$f" | tail -1)
  conv=$(awk "BEGIN{print (${best:-0} >= 99.0) ? 1 : 0}")
  echo "  $n best=${best:-?} converged=$conv"
  [ "$conv" = "1" ] || FAIL=1
done

if [ "$FAIL" = "1" ]; then
  echo "[$(date '+%F %T')] failures present -> launching l6 deepened batch"
  BATCH_RUN_GPUS=0,1,3,4,5,6 BATCH_RUN_IDLE_MEM_MB=4096 setsid nohup \
    "$PYBIN" -u src/batch_run.py \
    --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
    "$J6" > "logs/run_variant_o2_fillblank_noaux_randpos_d512l6_P127_len3.log" 2>&1 < /dev/null &
  sleep 15
  pgrep -f "batch_run.py .*$J6" > /dev/null 2>&1 \
    && echo "[$(date '+%F %T')] LAUNCHED_L6" || { echo "[$(date '+%F %T')] L6_START_FAILED"; exit 1; }
  # 二级条件监视器：l6 跑完仍失败则自动上 l8（detached，独立存活）
  setsid nohup bash scripts/cond_deepen_fillblank_l6_to_l8.sh \
    > logs/cond_deepen_l6_to_l8.log 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] l6->l8 watcher armed (logs/cond_deepen_l6_to_l8.log)"
else
  echo "[$(date '+%F %T')] ALL_CONVERGED no deepen needed"
fi
