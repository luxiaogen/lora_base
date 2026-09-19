#!/usr/bin/env bash
# Run from the repository root. No data-path override and no environment install.
set -uo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
CL_PYTHON=${CL_PYTHON:-$PYTHON_BIN}
SD_PYTHON=${SD_PYTHON:-$PYTHON_BIN}
LOG_DIR=logs/shell_logs/dualmask_transfer_3090
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=0
mkdir -p "$LOG_DIR"

# Download exact upstream sources once. This never alters the local JSON.
for host in cl sd; do
    "$PYTHON_BIN" integrations/dualmask_transfer/run.py --host "$host" --setup || exit 1
done
"$PYTHON_BIN" -m unittest test.test_dualmask_transfer || exit 1

# Preflight both environments before spending a night on training.
for interpreter in "$CL_PYTHON" "$SD_PYTHON"; do
    "$interpreter" -c 'import torch, torchvision, timm, scipy, sklearn, einops, easydict; assert torch.cuda.is_available(), "CUDA required"' || exit 1
done

for host in cl sd; do
    interpreter=$CL_PYTHON
    [[ "$host" == sd ]] && interpreter=$SD_PYTHON
    # A short two-task GPU smoke for BOTH arms is mandatory before the pair.
    healthy=1
    for mode in off on; do
        if ! "$interpreter" integrations/dualmask_transfer/run.py --host "$host" --mode "$mode" --smoke 2>&1 | tee "$LOG_DIR/${host}_${mode}_smoke_${STAMP}.log"; then
            healthy=0
            FAILED=1
        fi
    done
    if [[ "$healthy" != 1 ]]; then
        echo "SKIP full $host pair: smoke failed"
        continue
    fi
    [[ "${SMOKE_ONLY:-0}" == 1 ]] && continue
    for mode in off on; do
        echo "Starting $host mode=$mode ImageNet-R T10 seed1993 from scratch"
        if ! "$interpreter" integrations/dualmask_transfer/run.py --host "$host" --mode "$mode" 2>&1 | tee "$LOG_DIR/${host}_${mode}_t10_${STAMP}.log"; then
            FAILED=1
        fi
    done
done
echo "Finished transfer sweep; FAILED=$FAILED; logs=$LOG_DIR"
exit "$FAILED"
