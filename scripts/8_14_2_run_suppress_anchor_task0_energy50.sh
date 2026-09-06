#!/usr/bin/env bash
set -uo pipefail

LOG_DIR=logs/shell_logs/suppress_anchor_task0_energy50_3datasets_3seeds
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FAILED=0
mkdir -p "$LOG_DIR"

#echo "============================================================"
#echo "Starting cub10_suppress_anchor_task0_w10_energy50_seed1993"
#echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
#echo "Log: $LOG_DIR/cub10_suppress_anchor_task0_w10_energy50_seed1993_${TIMESTAMP}.log"
#echo "============================================================"
#if
#    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
#        --set 'seed=[1993]' \
#        --set prefix=cub10_suppress_anchor_task0_w10_energy50_seed1993 \
#        --set init_epoch=20 \
#        --set epochs=20 \
#        --set ca=true \
#        --set dual_mask_task0_gate_mode=unmasked \
#        --set dual_mask_conflict_energy_adaptive=true \
#        --set dual_mask_conflict_energy_ratio_floor=false \
#        --set dual_mask_conflict_ratio=0.1 \
#        --set dual_mask_conflict_strength=0.5 \
#        --set dual_mask_conflict_old_overlap_adaptive=true \
#        --set dual_mask_conflict_reg_enabled=false \
#        --set dual_mask_reg_weight=0.01 \
#        --set dual_mask_private_conflict_mode=global \
#        --set dual_mask_conflict_merge_mode=suppress \
#        --set dual_mask_anchor_reg_enabled=true \
#        --set dual_mask_anchor_reg_weight=10.0 \
#        --set dual_mask_anchor_reg_task0_only=true \
#        --set dual_mask_selective_anchor_enabled=false \
#        --set dual_mask_functional_merge_calibration=false \
#        --set dual_mask_safe_residual_enabled=false \
#        --set dual_mask_track_w0_metrics=true \
#        --set dual_mask_vis=false \
#        --set experiment_tracker=wandb \
#        --set wandb_project=LoDA_ICML2026 \
#        --set wandb_mode=online \
#        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
#        --set rank=32 \
#        --set wandb_tags=cub10,suppress,energy50,anchor_task0,w10,night,multiseed \
#        2>&1 | tee "$LOG_DIR/cub10_suppress_anchor_task0_w10_energy50_seed1993_${TIMESTAMP}.log"
#then
#    echo "PASS cub10_suppress_anchor_task0_w10_energy50_seed1993"
#else
#    echo "FAIL cub10_suppress_anchor_task0_w10_energy50_seed1993"
#    FAILED=1
#fi

#echo "============================================================"
#echo "Starting cub10_suppress_anchor_task0_w10_energy50_seed1996"
#echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
#echo "Log: $LOG_DIR/cub10_suppress_anchor_task0_w10_energy50_seed1996_${TIMESTAMP}.log"
#echo "============================================================"
#if
#    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
#        --set 'seed=[1996]' \
#        --set prefix=cub10_suppress_anchor_task0_w10_energy50_seed1996 \
#        --set init_epoch=20 \
#        --set epochs=20 \
#        --set ca=true \
#        --set dual_mask_task0_gate_mode=unmasked \
#        --set dual_mask_conflict_energy_adaptive=true \
#        --set dual_mask_conflict_energy_ratio_floor=false \
#        --set dual_mask_conflict_ratio=0.1 \
#        --set dual_mask_conflict_strength=0.5 \
#        --set dual_mask_conflict_old_overlap_adaptive=true \
#        --set dual_mask_conflict_reg_enabled=false \
#        --set dual_mask_reg_weight=0.01 \
#        --set dual_mask_private_conflict_mode=global \
#        --set dual_mask_conflict_merge_mode=suppress \
#        --set dual_mask_anchor_reg_enabled=true \
#        --set dual_mask_anchor_reg_weight=10.0 \
#        --set dual_mask_anchor_reg_task0_only=true \
#        --set dual_mask_selective_anchor_enabled=false \
#        --set dual_mask_functional_merge_calibration=false \
#        --set dual_mask_safe_residual_enabled=false \
#        --set dual_mask_track_w0_metrics=true \
#        --set dual_mask_vis=false \
#        --set experiment_tracker=wandb \
#        --set wandb_project=LoDA_ICML2026 \
#        --set wandb_mode=online \
#        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
#        --set rank=32 \
#        --set wandb_tags=cub10,suppress,energy50,anchor_task0,w10,night,multiseed \
#        2>&1 | tee "$LOG_DIR/cub10_suppress_anchor_task0_w10_energy50_seed1996_${TIMESTAMP}.log"
#then
#    echo "PASS cub10_suppress_anchor_task0_w10_energy50_seed1996"
#else
#    echo "FAIL cub10_suppress_anchor_task0_w10_energy50_seed1996"
#    FAILED=1
#fi

#echo "============================================================"
#echo "Starting cub10_suppress_anchor_task0_w10_energy50_seed1997"
#echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
#echo "Log: $LOG_DIR/cub10_suppress_anchor_task0_w10_energy50_seed1997_${TIMESTAMP}.log"
#echo "============================================================"
#if
#    python main.py --config ideas/dual_mask_branch/configs/cub10.json \
#        --set 'seed=[1997]' \
#        --set prefix=cub10_suppress_anchor_task0_w10_energy50_seed1997 \
#        --set init_epoch=20 \
#        --set epochs=20 \
#        --set ca=true \
#        --set dual_mask_task0_gate_mode=unmasked \
#        --set dual_mask_conflict_energy_adaptive=true \
#        --set dual_mask_conflict_energy_ratio_floor=false \
#        --set dual_mask_conflict_ratio=0.1 \
#        --set dual_mask_conflict_strength=0.5 \
#        --set dual_mask_conflict_old_overlap_adaptive=true \
#        --set dual_mask_conflict_reg_enabled=false \
#        --set dual_mask_reg_weight=0.01 \
#        --set dual_mask_private_conflict_mode=global \
#        --set dual_mask_conflict_merge_mode=suppress \
#        --set dual_mask_anchor_reg_enabled=true \
#        --set dual_mask_anchor_reg_weight=10.0 \
#        --set dual_mask_anchor_reg_task0_only=true \
#        --set dual_mask_selective_anchor_enabled=false \
#        --set dual_mask_functional_merge_calibration=false \
#        --set dual_mask_safe_residual_enabled=false \
#        --set dual_mask_track_w0_metrics=true \
#        --set dual_mask_vis=false \
#        --set experiment_tracker=wandb \
#        --set wandb_project=LoDA_ICML2026 \
#        --set wandb_mode=online \
#        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
#        --set rank=32 \
#        --set wandb_tags=cub10,suppress,energy50,anchor_task0,w10,night,multiseed \
#        2>&1 | tee "$LOG_DIR/cub10_suppress_anchor_task0_w10_energy50_seed1997_${TIMESTAMP}.log"
#then
#    echo "PASS cub10_suppress_anchor_task0_w10_energy50_seed1997"
#else
#    echo "FAIL cub10_suppress_anchor_task0_w10_energy50_seed1997"
#    FAILED=1
#fi

echo "============================================================"
echo "Starting imgr10_suppress_anchor_task0_w10_energy50_seed1993"
echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
echo "Log: $LOG_DIR/imgr10_suppress_anchor_task0_w10_energy50_seed1993_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1993]' \
        --set prefix=imgr10_suppress_anchor_task0_w10_energy50_seed1993 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
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
        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
        --set rank=64 \
        --set wandb_tags=imgr10,suppress,energy50,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_anchor_task0_w10_energy50_seed1993_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_anchor_task0_w10_energy50_seed1993"
else
    echo "FAIL imgr10_suppress_anchor_task0_w10_energy50_seed1993"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_anchor_task0_w10_energy50_seed1996"
echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
echo "Log: $LOG_DIR/imgr10_suppress_anchor_task0_w10_energy50_seed1996_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1996]' \
        --set prefix=imgr10_suppress_anchor_task0_w10_energy50_seed1996 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
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
        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
        --set rank=64 \
        --set wandb_tags=imgr10,suppress,energy50,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_anchor_task0_w10_energy50_seed1996_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_anchor_task0_w10_energy50_seed1996"
else
    echo "FAIL imgr10_suppress_anchor_task0_w10_energy50_seed1996"
    FAILED=1
fi

echo "============================================================"
echo "Starting imgr10_suppress_anchor_task0_w10_energy50_seed1997"
echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
echo "Log: $LOG_DIR/imgr10_suppress_anchor_task0_w10_energy50_seed1997_${TIMESTAMP}.log"
echo "============================================================"
if
    python main.py --config ideas/dual_mask_branch/configs/imgr10.json \
        --set 'seed=[1997]' \
        --set prefix=imgr10_suppress_anchor_task0_w10_energy50_seed1997 \
        --set init_epoch=20 \
        --set epochs=20 \
        --set ca=true \
        --set dual_mask_task0_gate_mode=unmasked \
        --set dual_mask_conflict_energy_adaptive=true \
        --set dual_mask_conflict_energy_ratio_floor=false \
        --set dual_mask_conflict_ratio=0.1 \
        --set dual_mask_conflict_strength=0.5 \
        --set dual_mask_conflict_old_overlap_adaptive=true \
        --set dual_mask_conflict_reg_enabled=false \
        --set dual_mask_reg_weight=0.01 \
        --set dual_mask_private_conflict_mode=global \
        --set dual_mask_conflict_merge_mode=suppress \
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
        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
        --set rank=64 \
        --set wandb_tags=imgr10,suppress,energy50,anchor_task0,w10,night,multiseed \
        2>&1 | tee "$LOG_DIR/imgr10_suppress_anchor_task0_w10_energy50_seed1997_${TIMESTAMP}.log"
then
    echo "PASS imgr10_suppress_anchor_task0_w10_energy50_seed1997"
else
    echo "FAIL imgr10_suppress_anchor_task0_w10_energy50_seed1997"
    FAILED=1
fi

#echo "============================================================"
#echo "Starting imga10_suppress_anchor_task0_w10_energy50_seed1993"
#echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
#echo "Log: $LOG_DIR/imga10_suppress_anchor_task0_w10_energy50_seed1993_${TIMESTAMP}.log"
#echo "============================================================"
#if
#    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
#        --set 'seed=[1993]' \
#        --set prefix=imga10_suppress_anchor_task0_w10_energy50_seed1993 \
#        --set init_epoch=20 \
#        --set epochs=20 \
#        --set ca=true \
#        --set dual_mask_task0_gate_mode=unmasked \
#        --set dual_mask_conflict_energy_adaptive=true \
#        --set dual_mask_conflict_energy_ratio_floor=false \
#        --set dual_mask_conflict_ratio=0.1 \
#        --set dual_mask_conflict_strength=0.5 \
#        --set dual_mask_conflict_old_overlap_adaptive=true \
#        --set dual_mask_conflict_reg_enabled=false \
#        --set dual_mask_reg_weight=0.01 \
#        --set dual_mask_private_conflict_mode=global \
#        --set dual_mask_conflict_merge_mode=suppress \
#        --set dual_mask_anchor_reg_enabled=true \
#        --set dual_mask_anchor_reg_weight=10.0 \
#        --set dual_mask_anchor_reg_task0_only=true \
#        --set dual_mask_selective_anchor_enabled=false \
#        --set dual_mask_functional_merge_calibration=false \
#        --set dual_mask_safe_residual_enabled=false \
#        --set dual_mask_track_w0_metrics=true \
#        --set dual_mask_vis=false \
#        --set experiment_tracker=wandb \
#        --set wandb_project=LoDA_ICML2026 \
#        --set wandb_mode=online \
#        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
#        --set rank=32 \
#        --set wandb_tags=imga10,suppress,energy50,anchor_task0,w10,night,multiseed \
#        2>&1 | tee "$LOG_DIR/imga10_suppress_anchor_task0_w10_energy50_seed1993_${TIMESTAMP}.log"
#then
#    echo "PASS imga10_suppress_anchor_task0_w10_energy50_seed1993"
#else
#    echo "FAIL imga10_suppress_anchor_task0_w10_energy50_seed1993"
#    FAILED=1
#fi
#
#echo "============================================================"
#echo "Starting imga10_suppress_anchor_task0_w10_energy50_seed1996"
#echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
#echo "Log: $LOG_DIR/imga10_suppress_anchor_task0_w10_energy50_seed1996_${TIMESTAMP}.log"
#echo "============================================================"
#if
#    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
#        --set 'seed=[1996]' \
#        --set prefix=imga10_suppress_anchor_task0_w10_energy50_seed1996 \
#        --set init_epoch=20 \
#        --set epochs=20 \
#        --set ca=true \
#        --set dual_mask_task0_gate_mode=unmasked \
#        --set dual_mask_conflict_energy_adaptive=true \
#        --set dual_mask_conflict_energy_ratio_floor=false \
#        --set dual_mask_conflict_ratio=0.1 \
#        --set dual_mask_conflict_strength=0.5 \
#        --set dual_mask_conflict_old_overlap_adaptive=true \
#        --set dual_mask_conflict_reg_enabled=false \
#        --set dual_mask_reg_weight=0.01 \
#        --set dual_mask_private_conflict_mode=global \
#        --set dual_mask_conflict_merge_mode=suppress \
#        --set dual_mask_anchor_reg_enabled=true \
#        --set dual_mask_anchor_reg_weight=10.0 \
#        --set dual_mask_anchor_reg_task0_only=true \
#        --set dual_mask_selective_anchor_enabled=false \
#        --set dual_mask_functional_merge_calibration=false \
#        --set dual_mask_safe_residual_enabled=false \
#        --set dual_mask_track_w0_metrics=true \
#        --set dual_mask_vis=false \
#        --set experiment_tracker=wandb \
#        --set wandb_project=LoDA_ICML2026 \
#        --set wandb_mode=online \
#        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
#        --set rank=32 \
#        --set wandb_tags=imga10,suppress,energy50,anchor_task0,w10,night,multiseed \
#        2>&1 | tee "$LOG_DIR/imga10_suppress_anchor_task0_w10_energy50_seed1996_${TIMESTAMP}.log"
#then
#    echo "PASS imga10_suppress_anchor_task0_w10_energy50_seed1996"
#else
#    echo "FAIL imga10_suppress_anchor_task0_w10_energy50_seed1996"
#    FAILED=1
#fi
#
#echo "============================================================"
#echo "Starting imga10_suppress_anchor_task0_w10_energy50_seed1997"
#echo "Changed: Best current protocol: suppress merge, Task-0 W0 anchor w10, adaptive coordinate set covering 50 percent conflict energy"
#echo "Log: $LOG_DIR/imga10_suppress_anchor_task0_w10_energy50_seed1997_${TIMESTAMP}.log"
#echo "============================================================"
#if
#    python main.py --config ideas/dual_mask_branch/configs/imga10.json \
#        --set 'seed=[1997]' \
#        --set prefix=imga10_suppress_anchor_task0_w10_energy50_seed1997 \
#        --set init_epoch=20 \
#        --set epochs=20 \
#        --set ca=true \
#        --set dual_mask_task0_gate_mode=unmasked \
#        --set dual_mask_conflict_energy_adaptive=true \
#        --set dual_mask_conflict_energy_ratio_floor=false \
#        --set dual_mask_conflict_ratio=0.1 \
#        --set dual_mask_conflict_strength=0.5 \
#        --set dual_mask_conflict_old_overlap_adaptive=true \
#        --set dual_mask_conflict_reg_enabled=false \
#        --set dual_mask_reg_weight=0.01 \
#        --set dual_mask_private_conflict_mode=global \
#        --set dual_mask_conflict_merge_mode=suppress \
#        --set dual_mask_anchor_reg_enabled=true \
#        --set dual_mask_anchor_reg_weight=10.0 \
#        --set dual_mask_anchor_reg_task0_only=true \
#        --set dual_mask_selective_anchor_enabled=false \
#        --set dual_mask_functional_merge_calibration=false \
#        --set dual_mask_safe_residual_enabled=false \
#        --set dual_mask_track_w0_metrics=true \
#        --set dual_mask_vis=false \
#        --set experiment_tracker=wandb \
#        --set wandb_project=LoDA_ICML2026 \
#        --set wandb_mode=online \
#        --set wandb_group=suppress_anchor_task0_energy50_3datasets_3seeds \
#        --set rank=32 \
#        --set wandb_tags=imga10,suppress,energy50,anchor_task0,w10,night,multiseed \
#        2>&1 | tee "$LOG_DIR/imga10_suppress_anchor_task0_w10_energy50_seed1997_${TIMESTAMP}.log"
#then
#    echo "PASS imga10_suppress_anchor_task0_w10_energy50_seed1997"
#else
#    echo "FAIL imga10_suppress_anchor_task0_w10_energy50_seed1997"
#    FAILED=1
#fi

echo "============================================================"
echo "Finished 3 runs for suppress_anchor_task0_energy50_3datasets_3seeds; FAILED=$FAILED"
echo "Logs: $LOG_DIR"
echo "============================================================"
exit $FAILED