#!/bin/bash
# 在空闲 GPU 上 benchmark 不同 BATCH_SIZE 的每 epoch 耗时与峰值显存
# 用法: bash bench_bs.sh <gpu_id>
cd "$(dirname "$0")"
GPU="${1:-2}"
BASE_CFG=bench_base.json
OUT=bench_bs_results.txt

run_bench() {
    local bs=$1 epochs=$2
    local cfg="config_bench_bs${bs}.json"
    python3 - "$BASE_CFG" "$cfg" "$bs" "$epochs" <<'EOF'
import json, sys
src, dst, bs, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(src))
d['main']['BATCH_SIZE'] = bs
d['main']['EPOCHS'] = epochs
d['main']['EVAL_INTERVAL'] = 20
d['main']['EARLY_STOP_ACCURACY'] = 2.0      # 禁用早停
d['main']['EARLY_STOP_NO_IMPROVE'] = 10**9
d['main']['SAVE_PATH'] = f'/tmp/bench_bs{bs}.pth'
json.dump(d, open(dst, 'w'), indent=2)
EOF
    CUDA_VISIBLE_DEVICES=$GPU python src/core.py "$cfg" > /tmp/bench_bs${bs}.log 2>&1 &
    local cpid=$!
    (
        max=0
        while kill -0 $cpid 2>/dev/null; do
            m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
            [ "$m" -gt "$max" ] && max=$m
            sleep 2
        done
        echo "$max" > /tmp/bench_maxmem_$bs
    ) &
    local poller=$!
    local t0=$(date +%s)
    wait $cpid; local rc=$?
    local t1=$(date +%s)
    wait $poller   # 轮询循环会自行退出并写入峰值
    local peak=$(cat /tmp/bench_maxmem_$bs 2>/dev/null || echo NA)
    echo "bs=$bs epochs=$epochs wall=$((t1-t0))s peak_mem=${peak}MiB rc=$rc" | tee -a "$OUT"
    rm -f "$cfg" /tmp/bench_bs${bs}.pth /tmp/bench_maxmem_$bs
}

for bs in 1024 2048 4096 8192; do
    run_bench $bs 200
done
echo "done" | tee -a "$OUT"
