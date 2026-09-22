#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
if [ "$#" -eq 0 ]; then
    set -- run --background
fi
exec bash "$REPO_DIR/bundles/action_online_300k_11runs_3gpu_6h_20260922/run.sh" "$@"
