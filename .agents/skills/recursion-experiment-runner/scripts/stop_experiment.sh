#!/bin/bash
# 技能3：在下一个存档点退出某条实验（不丢进度）。
# 对训练进程发 SIGTERM：训练循环在下一个 eval 点保存 <model>_resume.pth
# （完整训练状态）后以退出码 42 退出。编排器不受影响，队列照常前进。
# 用法: bash stop_experiment.sh <实验名唯一子串>
set -u
PAT="${1:?用法: bash stop_experiment.sh <实验名唯一子串>}"

mapfile -t pids < <(pgrep -f "core.py config_tmp_.*${PAT}")
if [ "${#pids[@]}" -eq 0 ]; then
  echo "没有匹配的运行中实验: $PAT"
  exit 1
fi
if [ "${#pids[@]}" -gt 1 ]; then
  echo "匹配到多个进程，请改用更长的唯一子串:"
  pgrep -af "core.py config_tmp_.*${PAT}"
  exit 1
fi
kill "${pids[0]}"
echo "已 SIGTERM pid=${pids[0]}（$PAT）"
echo "训练进程会在下一个 eval 点存盘退出；日志出现 '[STOP] Checkpoint saved' 即完成。"
echo "注意：EVAL_INTERVAL 非 1 时最长可能要等一个评估间隔。"
