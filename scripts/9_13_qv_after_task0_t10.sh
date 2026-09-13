#!/usr/bin/env bash
set -euo pipefail

# From repository root. Task0 trains QK; Task1-9 compare QKV against QV.
# Optional TASK0_CHECKPOINT must come from a Task0 QK run on this machine.
# lrun scripts/9_13_qv_after_task0_t10.sh ./logs/9_13_qv_after_task0_t10.log
MAX_TASKS=10 TASK0_QK=true exec bash scripts/9_13_qv_after_task0.sh "$@"
