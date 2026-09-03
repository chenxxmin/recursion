#!/bin/bash
# 链式多轮 runner：每轮重新生成 resume 配置（跳过已完成、自动接 checkpoint）
# 用法: bash scripts/chain_rounds.sh <config.json> <gpus> <rounds>
set -u
cd "$(dirname "$0")/.."
CONFIG="$1"; GPUS="$2"; ROUNDS="$3"
ts() { date '+%F %T'; }

# 可选隔离：CHAIN_DATA_BASE / CHAIN_MODEL_BASE 改变日志与模型的根目录
DIR_ARGS=()
[ -n "${CHAIN_DATA_BASE:-}" ] && DIR_ARGS+=(--base-dir "$CHAIN_DATA_BASE")
[ -n "${CHAIN_MODEL_BASE:-}" ] && DIR_ARGS+=(--model-base-dir "$CHAIN_MODEL_BASE")

for r in $(seq 1 "$ROUNDS"); do
    remaining=$(python3 scripts/prepare_chain_round.py "$CONFIG") || {
        echo "[$(ts)] 全部实验已完成，链结束"; exit 0; }
    echo "[$(ts)] === round $r/$ROUNDS: $CONFIG (剩余 $remaining 个实验, GPU $GPUS) ==="
    BATCH_RUN_GPUS=$GPUS python -u src/batch_run.py "${DIR_ARGS[@]}" "$CONFIG"
    echo "[$(ts)] round $r 结束（码 $?）"
done
echo "[$(ts)] $ROUNDS 轮跑完"
