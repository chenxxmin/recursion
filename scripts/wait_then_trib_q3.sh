#!/bin/bash
cd /mnt/workspace/hujiachen/recursion
while kill -0 "$1" 2>/dev/null; do sleep 60; done
while pgrep -f '[b]atch_run.py experiments/mixed_abc_quad_q3' >/dev/null; do sleep 30; done
sleep 15
bash scripts/start_queue.sh 3 experiments/trib39108_27_p127_300k_q3.json
