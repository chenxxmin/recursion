#!/bin/bash
cd /mnt/workspace/hujiachen/recursion
while kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 15
bash scripts/start_queue.sh 1 experiments/mixed_basic_quad_q1.json
