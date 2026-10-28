#!/bin/bash
# GPU 空闲监控分配器：每 60s 扫一次，把未启动的分片队列塞到空卡上。
# 用法: setsid nohup bash scripts/gpu_monitor_assign.sh <shard_glob> <state_file> [batch_run args...] &
# 规则：一张卡一个队列（沿用 start_queue.sh 的检查）；卡必须显存<100MiB、利用率0、无 compute 进程才算空。
# 所有分片都启动且没有存活队列后自动退出。
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
SHARD_GLOB="$1"; STATE="$2"; shift 2
[ -n "$SHARD_GLOB" ] && [ -n "$STATE" ] || { echo "usage: $0 <shard_glob> <state_file> [batch_run args...]" >&2; exit 2; }
touch "$STATE"

queue_on_gpu() {  # $1=gpu -> 0 if a batch_run queue is pinned to this gpu
  for pid in $(pgrep -f "[b]atch_run.py"); do
    env_gpus=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^BATCH_RUN_GPUS=' | cut -d= -f2)
    [ "$env_gpus" = "$1" ] && return 0
  done
  return 1
}

gpu_idle() {  # $1=gpu -> 0 if fully idle
  read -r mem util <<< "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ' | tr ',' ' ')"
  [ -z "$mem" ] && return 1
  [ "$mem" -lt 100 ] && [ "$util" -eq 0 ] || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" 2>/dev/null)" ]
}

while true; do
  mapfile -t shards < <(ls $SHARD_GLOB 2>/dev/null)
  remaining=()
  for s in "${shards[@]}"; do grep -qxF "$s" "$STATE" || remaining+=("$s"); done
  if [ "${#remaining[@]}" -eq 0 ] && ! pgrep -f "[b]atch_run.py" >/dev/null; then
    echo "[$(date +%H:%M:%S)] all shards finished, monitor exit"
    break
  fi
  for gpu in $GPU_IDS; do
    [ "${#remaining[@]}" -eq 0 ] && break
    queue_on_gpu "$gpu" && continue
    if gpu_idle "$gpu"; then
      next="${remaining[0]}"
      echo "[$(date +%H:%M:%S)] GPU $gpu idle -> start shard $next"
      bash scripts/start_queue.sh "$gpu" "$next" "$@" && echo "$next" >> "$STATE"
      remaining=("${remaining[@]:1}")
      sleep 5  # 让队列进程起来，避免下轮重复判空
    fi
  done
  sleep 60
done
