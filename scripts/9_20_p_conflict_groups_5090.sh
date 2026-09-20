#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root. Do not change or override the JSON data path.
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=${DRY_RUN:-0}
if [[ ! -f main.py || ! -f exps/dlora/imgr10.json ]]; then
    echo "Run from the repository root." >&2
    exit 1
fi
if [[ "$DRY_RUN" != 1 ]]; then
    git rev-parse HEAD
    "$PYTHON_BIN" -m unittest test.test_p_conflict_groups test.test_p_region_diagnostic test.test_dual_mask_core -q || exit 1
    [[ "${CHECK_ONLY:-0}" == 1 ]] && exit 0
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import torch
config = json.load(open('exps/dlora/imgr10.json'))
root = Path(config['data_path'])
print('Dataset from JSON:', root)
assert config['dataset'] == 'ImageNet_R'
assert (root/'train').is_dir() and (root/'test').is_dir(), 'Existing train/test required; no auto-splitting'
assert torch.cuda.is_available(), 'Activate the CUDA environment'
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__)
PY
    [[ $? == 0 ]] || exit 1
fi
run_python() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRYRUN python '; printf '%q ' "$@"; printf '\n'
    else
        "$PYTHON_BIN" -u "$@"
    fi
}
LOG_DIR=${LOG_DIR:-logs/shell_logs/p_conflict_groups_5090}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_groups_seed1993"
echo "Changed: Read-only layer QKV group removal; no gate selection"
echo "Log: $LOG_DIR/imgr10_groups_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    run_python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_groups_seed1993_${TIMESTAMP}" \
        --set init_cls=20 \
        --set increment=20 \
        --set total_sessions=10 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set rank=64 \
        --set ca=true \
        --set ca_epochs=5 \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_private_rank=0 \
        --set dual_mask_anchor_reg_enabled=true \
        --set dual_mask_anchor_reg_weight=10.0 \
        --set dual_mask_anchor_reg_task0_only=true \
        --set dual_mask_selective_anchor_enabled=false \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_safe_residual_enabled=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set dual_mask_p_region_train_mode=none \
        --set dual_mask_p_region_diagnostic=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_conflict_groups_5090 \
        --set dual_mask_p_conflict_group_diagnostic=true \
        2>&1 | tee "$LOG_DIR/imgr10_groups_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_groups_seed1993"
else
    echo "FAIL imgr10_groups_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 1 runs for p_conflict_groups_5090; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
