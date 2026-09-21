#!/bin/bash
cd /mnt/workspace/hujiachen/recursion
while kill -0 "$1" 2>/dev/null; do sleep 60; done
while pgrep -f '[b]atch_run.py experiments/mixed_abc_trib2_d2048l2h8' >/dev/null; do sleep 30; done
sleep 15
bash scripts/start_queue.sh 0 experiments/trib92123_21_p127_300k_q0.json
