#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

python -m unittest test.test_lori_a_overlap
python scripts/diagnose_lora_a_overlap.py \
    --din 768 \
    --rank 64 \
    --tasks 10 \
    --layers 12 \
    --branches S P \
    --seeds 1993 1996 1997 \
    --sweep-din 256 512 768 1024 2048 4096 \
    --sweep-layers 3 \
    --out logs/diagnostics/lori_a_overlap_d768_r64.json
