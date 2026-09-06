#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")"

LOG_DIR="logs/shell_logs/directional_conflict_imgr_imga"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GROUP="directional_conflict_energy50_${TIMESTAMP}"
SEEDS=(1993)
FAILURES=0

mkdir -p "$LOG_DIR"

run_experiment() {
    local name="$1"
    local config="$2"
    local seed="$3"
    local log_file="${LOG_DIR}/${name}_seed${seed}_${TIMESTAMP}.log"

    echo "============================================================"
    echo "Starting ${name}, seed=${seed}"
    echo "score_mode=directional, conflict_energy=50%, ratio_floor=10%"
    echo "Log: ${log_file}"
    echo "============================================================"

    if python main.py \
        --config "$config" \
        --set "seed=[${seed}]" \
        --set "prefix=${name}_directional_energy50" \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_conflict_score_mode=directional \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_merge_mode=suppress \
        --set dual_mask_task0_gate_mode=protect_only \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set "wandb_group=${GROUP}" \
        --set "wandb_tags=directional_conflict,energy50,${name},seed${seed}" \
        2>&1 | tee "$log_file"
    then
        echo "Finished ${name}, seed=${seed}"
    else
        echo "FAILED ${name}, seed=${seed}"
        FAILURES=$((FAILURES + 1))
    fi
}

for seed in "${SEEDS[@]}"; do
    run_experiment \
        imgr10 \
        exps/dlora/imgr10.json \
        "$seed"

    run_experiment \
        imga10 \
        exps/dlora/imga10.json \
        "$seed"
done

echo "============================================================"
echo "Finished directional ImageNet-R/A experiments"
echo "Failures: ${FAILURES}"
echo "W&B group: ${GROUP}"
echo "============================================================"

exit "$FAILURES"