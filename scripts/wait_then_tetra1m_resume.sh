#!/bin/bash
# 等 tetra 1m 首轮(PID $1, 6h 定时)结束后，校验 checkpoint 并在卡0 续跑 12h
cd /mnt/workspace/hujiachen/recursion
while kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 15
M=/mnt/workspace/hujiachen/models/tetranacci_a76b66c4d113_p127_d1024l2_1m
CKPT=$(/usr/bin/python3 - "$M" <<'PY'
import sys, torch, os
m = sys.argv[1]
for name in ('tetranacci_a76b66c4d113_d1024l2r4h4_P127_1m_seed0_resume.pth',
             'tetranacci_a76b66c4d113_d1024l2r4h4_P127_1m_seed0_latest.pth'):
    p = os.path.join(m, name)
    if os.path.exists(p):
        try:
            torch.load(p, map_location='cpu', weights_only=False)
            print(p); break
        except Exception as e:
            print(f"corrupt: {p}: {e}", file=sys.stderr)
PY
)
if [ -z "$CKPT" ]; then echo "no valid checkpoint, abort"; exit 1; fi
sed "s|__RESUME_PATH__|$CKPT|" experiments/tetranacci_a76b66c4d113_p127_d1024l2_1m_resume.json > experiments/tetranacci_a76b66c4d113_p127_d1024l2_1m_resume_filled.json
BATCH_RUN_GPUS=0 /usr/bin/python3 -u src/batch_run.py experiments/tetranacci_a76b66c4d113_p127_d1024l2_1m_resume_filled.json \
  > logs/run_q0_tetra76664113_1m_resume.log 2>&1
