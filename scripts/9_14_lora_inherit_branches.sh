#!/usr/bin/env bash
set -uo pipefail

# Run from the repository root. Dataset paths remain in local JSON configs.
[[ -f main.py ]] || { echo 'Run from the repository root.' >&2; exit 2; }
[[ $# -gt 0 ]] || { echo 'Usage: bash scripts/9_14_lora_inherit_branches.sh imgr10 [cub10 imga10]' >&2; exit 2; }
for dataset in "$@"; do
    case "$dataset" in imgr10|cub10|imga10) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
done
export PYTHONUNBUFFERED=1
DRY_RUN=${DRY_RUN:-0}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=logs/shell_logs/lora_inherit_branches
FAILED=0
RUNS=0

if [[ "$DRY_RUN" != 1 ]]; then
    PYTHONWARNINGS=ignore python -m unittest test.test_lora_inherit test.test_lora_inherit_script || exit 1
    python - "$@" <<'PY'
import json
import sys
from pathlib import Path
import torch
for name in sys.argv[1:]:
    root = Path(json.loads(Path('exps/dlora/' + name + '.json').read_text())['data_path'])
    print(name, 'data_path:', root)
    if not root.is_dir():
        raise SystemExit('Missing dataset directory; update data_path in the local config')
    if name == 'imgr10' and not all((root / split).is_dir() for split in ('train', 'test')):
        raise SystemExit('Existing ImageNet-R train/test split required')
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))
PY
    [[ $? -eq 0 ]] || exit 1
    git rev-parse --short HEAD
    mkdir -p "$LOG_DIR"
fi

common=(
    --set total_sessions=10 --set init_cls=20 --set increment=20 --set max_tasks=10
    --set use_slora=true --set use_plora=true
    --set init_epoch=20 --set epochs=20 --set ca=true --set ca_epochs=5
    --set lora_A_init=kaiming
    --set dual_mask_qv_after_task0=false --set dual_mask_task0_qk=false
    --set task0_checkpoint_resume= --set task0_checkpoint_save=
    --set dual_mask_private_rank=0
    --set dual_mask_competence_adaptive=true --set dual_mask_plasticity_adaptive=true
    --set dual_mask_protect_strength_mode=competence
    --set dual_mask_task0_gate_mode=unmasked
    --set dual_mask_anchor_reg_enabled=true --set dual_mask_anchor_reg_weight=10
    --set dual_mask_anchor_reg_task0_only=true
    --set dual_mask_conflict_energy_adaptive=true --set dual_mask_conflict_energy_ratio_floor=true
    --set dual_mask_conflict_ratio=0.1 --set dual_mask_conflict_strength=0.5
    --set dual_mask_conflict_old_overlap_adaptive=true
    --set dual_mask_private_conflict_mode=global --set dual_mask_conflict_merge_mode=suppress
    --set dual_mask_conflict_reg_enabled=false --set dual_mask_reg_weight=0.01
    --set dual_mask_selective_anchor_enabled=false --set dual_mask_functional_merge_calibration=false
    --set dual_mask_safe_residual_enabled=false --set dual_mask_track_w0_metrics=true
    --set dual_mask_ca_diagnostics=false --set dual_mask_vis=false
    --set experiment_tracker=wandb --set wandb_project=LoDA_ICML2026 --set wandb_mode=online
    --set wandb_group=lora_inherit_branches_t10
)

for dataset in "$@"; do
    rank=32
    p_gamma=0.75
    [[ "$dataset" != imgr10 ]] || rank=64
    [[ "$dataset" != imga10 ]] || p_gamma=1
    for seed in 1993 1996 1997; do
        for variant in fresh s_only p_only; do
            enabled=true
            case "$variant" in
                fresh) enabled=false; branches=both ;;
                s_only) branches=s ;;
                p_only) branches=p ;;
            esac
            run_name="${dataset}_${variant}_seed${seed}_${TIMESTAMP}"
            command=(python main.py --config "exps/dlora/${dataset}.json"
                --set "seed=[$seed]" --set "prefix=$run_name"
                "${common[@]}" --set "rank=$rank" --set slora_gamma=0.5 --set "plora_gamma=$p_gamma"
                --set "dual_mask_lora_inherit=$enabled" --set "dual_mask_lora_inherit_branches=$branches"
            )
            RUNS=$((RUNS + 1))
            if [[ "$DRY_RUN" == 1 ]]; then
                printf '%q ' "${command[@]}"
                printf '\n'
                continue
            fi
            echo "Starting $run_name; inheritance=$enabled branches=$branches; Task0-9 from scratch"
            if "${command[@]}" 2>&1 | tee "$LOG_DIR/$run_name.log"; then
                echo "PASS $run_name"
            else
                echo "FAIL $run_name"
                FAILED=1
            fi
        done
    done
done
echo "Finished $RUNS runs; FAILED=$FAILED; logs: $LOG_DIR"
exit "$FAILED"
