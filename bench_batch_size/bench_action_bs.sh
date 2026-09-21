#!/bin/bash
# Benchmark action+misslen2 l4 模型不同 BATCH_SIZE 的每 epoch 耗时与峰值显存
# 用法: bash bench_action_bs.sh <gpu_id>
cd "$(dirname "$0")/.."   # 项目根目录
GPU="${1:-0}"
BASE_CFG=bench_batch_size/bench_action_l4_base.json
OUT=bench_batch_size/bench_action_bs_results.txt
EPOCHS="${BENCH_EPOCHS:-20}"

run_bench() {
    local bs=$1
    local cfg="bench_batch_size/config_bench_action_bs${bs}.json"
    /usr/bin/python3 - "$BASE_CFG" "$cfg" "$bs" <<'EOF'
import json, sys
src, dst, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src))
d['main']['BATCH_SIZE'] = bs
d['main']['EPOCHS'] = int(__import__('os').environ.get('BENCH_EPOCHS', '20'))
d['main']['EVAL_INTERVAL'] = 10**9      # benchmark 期间不 eval
d['main']['EARLY_STOP_ACCURACY'] = 2.0  # 禁用早停
d['main']['EARLY_STOP_NO_IMPROVE'] = 10**9
d['main']['SAVE_PATH'] = f'/tmp/bench_action_bs{bs}.pth'
json.dump(d, open(dst, 'w'), indent=2)
EOF
    CUDA_VISIBLE_DEVICES=$GPU /usr/bin/python3 src/core.py "$cfg" > /tmp/bench_action_bs${bs}.log 2>&1 &
    local cpid=$!
    (
        max=0
        while kill -0 $cpid 2>/dev/null; do
            m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
            [ "$m" -gt "$max" ] && max=$m
            sleep 2
        done
        echo "$max" > /tmp/bench_action_maxmem_$bs
    ) &
    local poller=$!
    local t0=$(date +%s)
    wait $cpid; local rc=$?
    local t1=$(date +%s)
    wait $poller
    local peak=$(cat /tmp/bench_action_maxmem_$bs 2>/dev/null || echo NA)
    local per_epoch=$(( (t1-t0) / EPOCHS ))
    echo "bs=$bs epochs=$EPOCHS wall=$((t1-t0))s ~${per_epoch}s/epoch peak_mem=${peak}MiB rc=$rc" | tee -a "$OUT"
    rm -f "$cfg" /tmp/bench_action_bs${bs}.pth /tmp/bench_action_maxmem_$bs
}

for bs in "$@"; do :; done  # noop
if [ $# -gt 1 ]; then shift; BSS="$@"; else BSS="1024 2048 4096 8192 16384"; fi
for bs in $BSS; do
    run_bench $bs
done
echo "done" | tee -a "$OUT"
