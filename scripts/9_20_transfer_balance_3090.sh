#!/usr/bin/env bash
# Repository root; dataset path ONLY from local exps/dlora/imgr10.json.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN=${PYTHON_BIN:-python}
CL_PYTHON=${CL_PYTHON:-$PYTHON_BIN}
SD_PYTHON=${SD_PYTHON:-$PYTHON_BIN}
LOG_DIR=logs/shell_logs/transfer_balance_3090
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=0
mkdir -p "$LOG_DIR"
git rev-parse HEAD
for host in cl sd; do
    "$PYTHON_BIN" integrations/dualmask_transfer/run.py --host "$host" --setup || exit 1
done
"$PYTHON_BIN" -m unittest test.test_dualmask_transfer test.test_transfer_balance -q || exit 1
for interpreter in "$CL_PYTHON" "$SD_PYTHON"; do
    "$interpreter" -c 'import torch, torchvision, timm, scipy, sklearn, einops, easydict; assert torch.cuda.is_available()' || exit 1
done
for host in cl sd; do
    if [[ "$host" == cl ]]; then
        interpreter=$CL_PYTHON
        options=(--host cl --mode on --cl-protection-strength 0.25)
    else
        interpreter=$SD_PYTHON
        options=(--host sd --mode off --sd-checkpoint --smoke-tasks 6)
    fi
    # SD smoke reaches the previous OOM stage, Task5, with the unchanged batch size.
    if ! "$interpreter" integrations/dualmask_transfer/run.py "${options[@]}" --smoke 2>&1 | tee "$LOG_DIR/${host}_smoke_${STAMP}.log"; then
        echo "SKIP $host full run: smoke failed"
        FAILED=1
        continue
    fi
    [[ "${SMOKE_ONLY:-0}" == 1 ]] && continue
    echo "Starting $host T10 seed1993 FROM SCRATCH; not resuming smoke weights"
    if ! "$interpreter" integrations/dualmask_transfer/run.py "${options[@]}" 2>&1 | tee "$LOG_DIR/${host}_t10_${STAMP}.log"; then
        FAILED=1
    fi
done
echo "Finished CL weak protection / SD off completion; FAILED=$FAILED"
exit "$FAILED"
