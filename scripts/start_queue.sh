#!/bin/bash
# 启动一个单卡队列：一张 GPU 对应一个 batch_run 编排器，顺序跑完给定 JSON 里的实验。
# 用法: bash scripts/start_queue.sh <gpu_id> <experiments.json> [batch_run 额外参数...]
#   gpu_id 只允许 0-3（本机硬性规定，4-7 禁用）
#   默认 BATCH_SIZE=512（src/config.json），bench 结论：与 1024 同速、显存 26.5GB
# 示例:
#   bash scripts/start_queue.sh 0 experiments/v2/my_batch.json
#   bash scripts/start_queue.sh 1 experiments/v2/my_batch_v2.json \
#       --base-dir /mnt/workspace/hujiachen/recursion_results_v2 \
#       --model-base-dir /mnt/workspace/hujiachen/models_v2
set -e
cd "$(dirname "$0")/.."

GPU="$1"; CONFIG="$2"; shift 2 || true

case "$GPU" in
    0|1|2|3) ;;
    *) echo "ERROR: gpu_id 只允许 0-3（本机硬性规定，GPU 4-7 禁用）" >&2; exit 2 ;;
esac
[ -f "$CONFIG" ] || { echo "ERROR: 配置不存在: $CONFIG" >&2; exit 2; }

# 同一张卡只允许一个队列编排器
for pid in $(pgrep -f "[b]atch_run.py"); do
    env_gpus=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^BATCH_RUN_GPUS=' | cut -d= -f2)
    if [ "$env_gpus" = "$GPU" ]; then
        echo "ERROR: GPU $GPU 已有队列在跑 (PID $pid)。先停掉或换卡。" >&2; exit 1
    fi
done

BATCH=$(basename "$CONFIG" .json)
mkdir -p logs
LOG="logs/run_q${GPU}_${BATCH}.log"

echo "GPU $GPU 队列启动: $CONFIG -> $LOG"
setsid nohup env BATCH_RUN_GPUS=$GPU BATCH_RUN_IDLE_MEM_MB=4096 \
    /usr/bin/python3 -u src/batch_run.py "$@" "$CONFIG" \
    > "$LOG" 2>&1 < /dev/null &
echo "orchestrator PID $!"
