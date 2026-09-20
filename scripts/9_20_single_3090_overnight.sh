#!/usr/bin/env bash
set -uo pipefail

if [[ ! -f main.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

FAILED=0

echo "Phase 1/2: 3 LoRI-style importance runs (seed 1993)"
if ! bash scripts/9_20_lori_style_importance_3datasets_seed1993.sh; then
    FAILED=1
fi

echo "Phase 2/2: 9 DualMask-off runs (3 datasets x 3 seeds)"
if ! bash scripts/9_20_dualmask_off_3datasets_3seeds.sh; then
    FAILED=1
fi

echo "============================================================"
echo "Finished single-3090 overnight queue; FAILED=$FAILED"
echo "============================================================"
exit "$FAILED"
