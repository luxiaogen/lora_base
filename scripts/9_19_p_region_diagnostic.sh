#!/usr/bin/env bash
set -uo pipefail

# Run from repository root, including through lrun. Never override JSON data_path.
if [[ ! -f main.py || ! -f exps/dlora/imgr10.json ]]; then
    echo "Run from the repository root." >&2
    exit 1
fi
git rev-parse HEAD
python -m unittest test.test_p_region_diagnostic test.test_dual_mask_core test.test_ca_diagnostics -q || exit 1
python -c 'import json; from pathlib import Path; p=Path(json.load(open("exps/dlora/imgr10.json"))["data_path"]); print("Dataset from JSON:",p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Existing train/test required; no auto-splitting"' || exit 1
LOG_DIR=logs/shell_logs/p_region_102b65c
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting imgr10_region_diagnostic_seed1993"
echo "Changed: Read-only C vs norm-matched U, current-task contribution only"
echo "Log: $LOG_DIR/imgr10_region_diagnostic_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config exps/dlora/imgr10.json \
        --set 'seed=[1993]' \
        --set "prefix=imgr10_region_diagnostic_seed1993_${TIMESTAMP}" \
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
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=p_region_102b65c \
        --set dual_mask_p_region_diagnostic=true \
        2>&1 | tee "$LOG_DIR/imgr10_region_diagnostic_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_region_diagnostic_seed1993"
else
    echo "FAIL imgr10_region_diagnostic_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Finished 1 runs for p_region_102b65c; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
