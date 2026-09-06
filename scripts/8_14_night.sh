#!/usr/bin/env bash
set -uo pipefail

FAILED=0

echo "============================================================"
echo "Phase 1/2: main 3-dataset x 3-seed sweep (9 runs)"
echo "============================================================"
if bash scripts/8_14_2_run_suppress_anchor_task0_energy50.sh; then
    echo "PASS phase 1: main sweep"
else
    echo "FAIL phase 1: main sweep"
    FAILED=1
fi

echo "============================================================"
echo "Phase 2/2: ImageNet-R seed-1993 matched validation (2 runs)"
echo "============================================================"
if bash scripts/8_14_3_run_anchor_task0_energy50_val_imgr.sh; then
    echo "PASS phase 2: matched validation"
else
    echo "FAIL phase 2: matched validation"
    FAILED=1
fi

echo "============================================================"
echo "Finished 11 overnight runs; FAILED=$FAILED"
echo "============================================================"
exit "$FAILED"