#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${EXPERIMENT_PYTHON:-python3}"
# 每个实验单独计时，单位小时；null=不限。达到时限后在 eval 点保存退出。
# 定时由训练循环 MAX_TRAIN_HOURS 执行，不改变 EPOCHS，不使用强杀。
TIMERS_JSON='{"action_d1024l2r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h":6,"action_d1024l2r4h4_P127_N2_tr64_miss01len2_online300k_seed17996_6h":6,"action_d1024l2r4h4_P127_N2_tr64_miss01len3_online300k_seed17996_6h":6,"action_d1024l2r4h4_P127_N2_tr64_miss01len5_online300k_seed17996_6h":6,"action_d1024l2r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h":6,"action_d1024l4r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h":6,"action_d1024l4r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h":6,"action_d1024l6r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h":6,"action_d1024l6r4h4_P127_N2_tr64_miss01len2_online300k_seed17996_6h":6,"action_d1024l6r4h4_P127_N2_tr64_miss01len3_online300k_seed17996_6h":6,"action_d1024l6r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h":6}'
exec "$PYTHON_BIN" "$BUNDLE_DIR/runner.py" --bundle "$BUNDLE_DIR" --timers "$TIMERS_JSON" "$@"
