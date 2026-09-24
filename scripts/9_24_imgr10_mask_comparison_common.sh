#!/usr/bin/env bash
set -euo pipefail

MACHINE=${1:?Expected 3090 or 5090}
DRY_RUN=${2:-}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

case "$MACHINE" in
    3090) VARIANTS=(layer projection model) ;;
    5090) VARIANTS=(budget050 budget100 budget150 magnitude100) ;;
    *) echo "Unknown machine: $MACHINE" >&2; exit 2 ;;
esac

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    python -m unittest test.test_global_conflict_budget test.test_mask_comparison_script
    python -c 'import json, pathlib; c=json.load(open("exps/dlora/imgr10.json")); p=pathlib.Path(c["data_path"]); print("Dataset from JSON:", p); assert (p/"train").is_dir() and (p/"test").is_dir(), "Missing train/ or test/"; expected={"rank":64,"epochs":20,"init_epoch":20,"lora_A_init":"kaiming","dual_mask_importance":"svd","dual_mask_svd_energy_coverage":0.95,"dual_mask_task0_gate_mode":"unmasked","use_slora":True,"use_plora":True}; bad={k:(c.get(k),v) for k,v in expected.items() if c.get(k)!=v}; assert not bad, f"Unexpected base config: {bad}"'
fi

echo "Code revision: $(git rev-parse --short HEAD)"
git status --short
LOG_DIR="logs/shell_logs/imgr10_mask_comparison_${MACHINE}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for VARIANT in "${VARIANTS[@]}"; do
    GRANULARITY=layer
    BUDGET_MULTIPLIER=1.0
    SCORE_MODE=conflict
    case "$VARIANT" in
        projection|model) GRANULARITY=$VARIANT ;;
        budget050) BUDGET_MULTIPLIER=0.5 ;;
        budget150) BUDGET_MULTIPLIER=1.5 ;;
        magnitude100) SCORE_MODE=magnitude ;;
    esac
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
        --set dual_mask_conflict_reg_enabled=false
        --set dual_mask_conflict_energy_adaptive=true
        --set dual_mask_conflict_energy_ratio_floor=true
        --set dual_mask_conflict_strength=0.5
        --set dual_mask_conflict_ratio=0.1
        --set "dual_mask_conflict_granularity=${GRANULARITY}"
        --set "dual_mask_conflict_budget_multiplier=${BUDGET_MULTIPLIER}"
        --set "dual_mask_conflict_score_mode=${SCORE_MODE}"
        --set dual_mask_private_conflict_mode=global
        --set experiment_tracker=wandb
        --set wandb_project=LoDA_ICML2026
        --set wandb_mode=online
        --set "wandb_group=imgr10_mask_comparison_${MACHINE}"
        --set "wandb_tags=imgr10,t10,seed1993,ca5,no_conflict_reg,${VARIANT}"
    )
    COMMAND=(python main.py --config exps/dlora/imgr10.json "${OVERRIDES[@]}")
    echo "Starting $VARIANT on $MACHINE (ImageNet-R T10, seed1993, CA5)"
    printf '%q ' "${COMMAND[@]}"
    echo
    if [[ "$DRY_RUN" != "--dry-run" ]]; then
        mkdir -p "$LOG_DIR"
        "${COMMAND[@]}" 2>&1 | tee "$LOG_DIR/imgr10_${VARIANT}_seed1993_${TIMESTAMP}.log"
    fi
done
