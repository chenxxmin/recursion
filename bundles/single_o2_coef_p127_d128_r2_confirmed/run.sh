#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${EXPERIMENT_PYTHON:-python3}"
# 每个实验单独计时，单位小时；null=不限。达到时限后在 eval 点保存退出。
# 定时由训练循环 MAX_TRAIN_HOURS 执行，不改变 EPOCHS，不使用强杀。
TIMERS_JSON='{"single_o2_a100b7_d128l2r8h4_P127_seed17996":1,"single_o2_a100b7_d128l2r8h4_P127_seed18318":1,"single_o2_a121b55_d128l2r8h4_P127_seed17996":1,"single_o2_a121b55_d128l2r8h4_P127_seed18318":1,"single_o2_a14b24_d128l2r8h4_P127_seed17996":1,"single_o2_a14b24_d128l2r8h4_P127_seed18318":1,"single_o2_a20b48_d128l2r8h4_P127_seed17996":1,"single_o2_a20b48_d128l2r8h4_P127_seed18318":1}'
exec "$PYTHON_BIN" "$BUNDLE_DIR/runner.py" --bundle "$BUNDLE_DIR" --timers "$TIMERS_JSON" "$@"
