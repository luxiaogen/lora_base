#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"
LOG_DIR=logs/shell_logs/best_suppress_relocate_3datasets_3seeds
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting cub10_suppress_relocate_seed1993"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/cub10_suppress_relocate_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
        --set 'seed=[1993]' \
        --set prefix=cub10_suppress_relocate_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_general_ratio=0.4 \
        --set wandb_tags=best_baseline,suppress_relocate,cub10 \
        2>&1 | tee "$LOG_DIR/cub10_suppress_relocate_seed1993_${TIMESTAMP}.log"
then
    echo "PASS cub10_suppress_relocate_seed1993"
else
    echo "FAIL cub10_suppress_relocate_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_suppress_relocate_seed1996"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/cub10_suppress_relocate_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
        --set 'seed=[1996]' \
        --set prefix=cub10_suppress_relocate_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_general_ratio=0.4 \
        --set wandb_tags=best_baseline,suppress_relocate,cub10 \
        2>&1 | tee "$LOG_DIR/cub10_suppress_relocate_seed1996_${TIMESTAMP}.log"
then
    echo "PASS cub10_suppress_relocate_seed1996"
else
    echo "FAIL cub10_suppress_relocate_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting cub10_suppress_relocate_seed1997"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/cub10_suppress_relocate_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
        --set 'seed=[1997]' \
        --set prefix=cub10_suppress_relocate_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_general_ratio=0.4 \
        --set wandb_tags=best_baseline,suppress_relocate,cub10 \
        2>&1 | tee "$LOG_DIR/cub10_suppress_relocate_seed1997_${TIMESTAMP}.log"
then
    echo "PASS cub10_suppress_relocate_seed1997"
else
    echo "FAIL cub10_suppress_relocate_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_relocate_seed1993"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/imgr10_suppress_relocate_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_suppress_relocate_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_general_ratio=0.4 \
        --set wandb_tags=best_baseline,suppress_relocate,imgr10 \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_relocate_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_relocate_seed1993"
else
    echo "FAIL imgr10_suppress_relocate_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_relocate_seed1996"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/imgr10_suppress_relocate_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_suppress_relocate_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_general_ratio=0.4 \
        --set wandb_tags=best_baseline,suppress_relocate,imgr10 \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_relocate_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_relocate_seed1996"
else
    echo "FAIL imgr10_suppress_relocate_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_relocate_seed1997"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/imgr10_suppress_relocate_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_suppress_relocate_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=64 \
        --set slora_gamma=0.5 \
        --set plora_gamma=0.75 \
        --set dual_mask_general_ratio=0.4 \
        --set wandb_tags=best_baseline,suppress_relocate,imgr10 \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_relocate_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_relocate_seed1997"
else
    echo "FAIL imgr10_suppress_relocate_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_suppress_relocate_seed1993"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/imga10_suppress_relocate_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
        --set 'seed=[1993]' \
        --set prefix=imga10_suppress_relocate_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1.0 \
        --set dual_mask_general_ratio=0.3 \
        --set wandb_tags=best_baseline,suppress_relocate,imga10 \
        2>&1 | tee "$LOG_DIR/imga10_suppress_relocate_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imga10_suppress_relocate_seed1993"
else
    echo "FAIL imga10_suppress_relocate_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_suppress_relocate_seed1996"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/imga10_suppress_relocate_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
        --set 'seed=[1996]' \
        --set prefix=imga10_suppress_relocate_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1.0 \
        --set dual_mask_general_ratio=0.3 \
        --set wandb_tags=best_baseline,suppress_relocate,imga10 \
        2>&1 | tee "$LOG_DIR/imga10_suppress_relocate_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imga10_suppress_relocate_seed1996"
else
    echo "FAIL imga10_suppress_relocate_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imga10_suppress_relocate_seed1997"
echo "Changed: Reproduce the best-known suppress_relocate baseline; only dataset and seed change"
echo "Log: $LOG_DIR/imga10_suppress_relocate_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
        --set 'seed=[1997]' \
        --set prefix=imga10_suppress_relocate_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set ca_epochs=10 \
        --set dual_mask_importance=svd \
        --set dual_mask_svd_rank=768 \
        --set dual_mask_svd_energy_coverage=0.95 \
        --set dual_mask_competence_adaptive=true \
        --set dual_mask_plasticity_adaptive=true \
        --set dual_mask_protect_strength_mode=competence \
        --set dual_mask_competence_all_seen=false \
        --set dual_mask_competence_metric=accuracy \
        --set dual_mask_layerwise_ratio_mode=none \
        --set dual_mask_s_protect_enabled=true \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_energy_adaptive=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_merge_mode=suppress_relocate \
        --set dual_mask_relocation_steps=20 \
        --set dual_mask_relocation_lr=0.1 \
        --set dual_mask_relocation_vectors=64 \
        --set dual_mask_functional_merge_calibration=false \
        --set dual_mask_track_w0_metrics=true \
        --set dual_mask_vis=false \
        --set experiment_tracker=wandb \
        --set wandb_project=LoDA_ICML2026 \
        --set wandb_mode=online \
        --set wandb_group=best_suppress_relocate_3datasets_3seeds \
        --set rank=32 \
        --set slora_gamma=0.5 \
        --set plora_gamma=1.0 \
        --set dual_mask_general_ratio=0.3 \
        --set wandb_tags=best_baseline,suppress_relocate,imga10 \
        2>&1 | tee "$LOG_DIR/imga10_suppress_relocate_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imga10_suppress_relocate_seed1997"
else
    echo "FAIL imga10_suppress_relocate_seed1997"
    FAILED=1
fi

echo "============================================================"
echo "Finished 9 runs for best_suppress_relocate_3datasets_3seeds; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED
