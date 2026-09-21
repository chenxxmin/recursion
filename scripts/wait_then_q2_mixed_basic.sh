#!/bin/bash
cd /mnt/workspace/hujiachen/recursion
# 等 mixed_abc q2 队列最后一个训练进程退出，且旧编排器退出后再开新队列
while kill -0 "$1" 2>/dev/null; do sleep 60; done
while pgrep -f '[b]atch_run.py experiments/mixed_abc_quad_q2.json' >/dev/null; do sleep 30; done
sleep 15
bash scripts/start_queue.sh 2 experiments/mixed_basic_quad_q2.json
