#!/bin/bash
# 技能4：给正在运行的实验设置/取消临时定时。到点后按技能3的逻辑
# （SIGTERM → 下一个存档点存盘退出）自动停止该实验，队列照常前进。
#
# 用法:
#   bash timer.sh set <实验名唯一子串> <小时>
#   bash timer.sh cancel <实验名唯一子串>
#   bash timer.sh list
#
# 注意：本工具只管正在运行的实验；未开始的实验请直接在实验 JSON 里
# 配 MAX_TRAIN_HOURS（每个实验各自计时）。
set -u
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
TIMER_DIR="$REPO/logs/timers"
mkdir -p "$TIMER_DIR"

cmd="${1:?用法: timer.sh set|cancel|list ...}"
pat="${2:-}"

timer_file() { echo "$TIMER_DIR/$(echo "$1" | tr -c 'A-Za-z0-9_.' '_').timer"; }

case "$cmd" in
  set)
    hours="${3:?缺少小时数}"
    # 确认目标在跑且唯一
    mapfile -t pids < <(pgrep -f "[c]ore.py.*config_tmp_.*${pat}")
    [ "${#pids[@]}" -eq 1 ] || { echo "匹配进程数=${#pids[@]}（需要恰好 1 个）"; exit 1; }
    f="$(timer_file "$pat")"
    deadline=$(( $(date +%s) + $(python3 -c "print(int($hours*3600))") ))
    setsid bash -c '
      f="$1"; pat="$2"; deadline="$3"; repo="$4"
      while true; do
        sleep 30
        [ -f "$f" ] || exit 0                       # 被取消
        tp=$(pgrep -f "[c]ore.py.*config_tmp_.*${pat}" | head -1)
        [ -z "$tp" ] && { rm -f "$f"; exit 0; }     # 实验已自行结束
        if [ "$(date +%s)" -ge "$deadline" ]; then
          cnt=$(pgrep -fc "[c]ore.py.*config_tmp_.*${pat}")
          if [ "$cnt" = 1 ]; then
            kill "$tp"
            echo "$(date -u "+%F %T") timer fired: SIGTERM $tp ($pat)" >> "$repo/logs/timers/fired.log"
          else
            echo "$(date -u "+%F %T") timer FAILED (matches=$cnt): $pat" >> "$repo/logs/timers/fired.log"
          fi
          rm -f "$f"
          exit 0
        fi
      done
    ' _ "$f" "$pat" "$deadline" "$REPO" < /dev/null > /dev/null 2>&1 &
    wpid=$!
    printf '{"pattern": "%s", "deadline": %s, "watcher_pid": %s}\n' "$pat" "$deadline" "$wpid" > "$f"
    echo "定时已设: $pat 将在 $hours 小时后（$(date -d @$deadline '+%F %T')）于下一个存档点存盘退出"
    ;;
  cancel)
    found=0
    for f in "$TIMER_DIR"/*.timer; do
      [ -f "$f" ] || continue
      stored=$(python3 -c "import json;print(json.load(open('$f'))['pattern'])")
      case "$stored" in
        *"$pat"*)
          wpid=$(python3 -c "import json;print(json.load(open('$f'))['watcher_pid'])")
          kill "$wpid" 2>/dev/null
          rm -f "$f"
          echo "已取消 $stored 的定时"
          found=1;;
      esac
    done
    [ "$found" = 0 ] && { echo "没有找到匹配 '$pat' 的定时"; exit 1; }
    ;;
  list)
    found=0
    for f in "$TIMER_DIR"/*.timer; do
      [ -f "$f" ] || continue
      found=1
      python3 -c "
import json, time
d = json.load(open('$f'))
rem = max(0, d['deadline'] - time.time())
print(f\"{d['pattern']}: 剩余 {rem/3600:.2f}h (watcher pid {d['watcher_pid']})\")"
    done
    [ "$found" = 0 ] && echo "没有活动定时"
    ;;
  *)
    echo "未知命令: $cmd（set|cancel|list）"; exit 1;;
esac
