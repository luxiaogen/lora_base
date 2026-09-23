#!/usr/bin/env bash
set -euo pipefail

MACHINE=${1:?Expected 3090 or 5090}
DRY_RUN=${2:-}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

case "$MACHINE" in
    3090) VARIANTS=(layer plastic_norm_matched model) ;;
    5090) VARIANTS=(layer ratio005 ratio020) ;;
    *) echo "Unknown machine: $MACHINE" >&2; exit 2 ;;
esac

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    python -m unittest test.test_dual_mask_core test.test_global_conflict_budget test.test_lori_night_script
    python -c 'import json, pathlib; p=pathlib.Path(json.load(open("exps/dlora/imgr10.json"))["data_path"]); print("Dataset:", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"'
fi

echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR="logs/shell_logs/imgr10_lori_night_${MACHINE}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for VARIANT in "${VARIANTS[@]}"; do
    OVERRIDES=(
        --set 'seed=[1993]'
        --set "prefix=imgr10_${VARIANT}_seed1993_${MACHINE}"
        --set ca_epochs=5
        --set max_tasks=10
        --set total_sessions=10
        --set init_cls=20
        --set increment=20
        --set task0_checkpoint_resume=
        --set task0_checkpoint_save=
        --set dual_mask_conflict_reg_enabled=true
        --set dual_mask_conflict_energy_adaptive=true
        --set dual_mask_conflict_energy_ratio_floor=true
        --set dual_mask_conflict_strength=0.5
        --set dual_mask_conflict_ratio=0.1
        --set dual_mask_conflict_granularity=layer
        --set dual_mask_private_conflict_mode=global
        --set experiment_tracker=wandb
        --set wandb_project=LoDA_ICML2026
        --set wandb_mode=online
        --set "wandb_group=imgr10_lori_night_${MACHINE}"
        --set "wandb_tags=imgr10,t10,seed1993,ca5,conflict_reg,${VARIANT}"
    )
    case "$VARIANT" in
        plastic_norm_matched) OVERRIDES+=(--set dual_mask_private_conflict_mode=plastic_norm_matched) ;;
        model) OVERRIDES+=(--set dual_mask_conflict_granularity=model) ;;
        ratio005) OVERRIDES+=(--set dual_mask_conflict_ratio=0.05) ;;
        ratio020) OVERRIDES+=(--set dual_mask_conflict_ratio=0.20) ;;
    esac
    COMMAND=(python main.py --config exps/dlora/imgr10.json "${OVERRIDES[@]}")
    echo "Starting $VARIANT on $MACHINE (ImageNet-R T10, seed1993, CA5)"
    printf '%q ' "${COMMAND[@]}"
    echo
    if [[ "$DRY_RUN" != "--dry-run" ]]; then
        mkdir -p "$LOG_DIR"
        "${COMMAND[@]}" 2>&1 | tee "$LOG_DIR/imgr10_${VARIANT}_seed1993_${TIMESTAMP}.log"
    fi
done
