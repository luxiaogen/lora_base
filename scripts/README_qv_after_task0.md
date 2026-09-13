# Task0 QKV, Task1 onward QV

This isolated experiment starts from `102b65c`. It adds one switch:

```text
dual_mask_qv_after_task0=false  # original QKV baseline (default)
dual_mask_qv_after_task0=true   # Task0 QKV; Task1 onward QV for both S and P
```

Task0 still trains K. Later tasks hold its **Task0-merged projection weight** fixed;
this is not the experiment that freezes pretrained K from Task0 onward. Q updates
and changing input features can still change attention scores and K activations.

The implementation zeros K rows of the effective delta before DualMask gates,
including forward, regularization and merge. QKV tensor shapes, initialization
order, original full-matrix mask statistics, and regularization denominators are
retained. This screening tests projection restriction; it does **not** claim
reduced allocated parameters, optimizer storage, or training FLOPs.

## Run from the repository root

Activate the server's training environment. Set `data_path` only in the server's
local `exps/dlora/imgr10.json`; the script does not override it or change directory.

```bash
lrun scripts/9_13_qv_after_task0.sh ./logs/9_13_qv_after_task0.log
```

The script runs these stages sequentially on each machine:

1. Train/evaluate/merge Task0 once and save a snapshot.
2. Restore that snapshot; train Task1–2 with QKV.
3. Restore the same snapshot and random states; train Task1–2 with QV.

Each invocation has a unique output directory under `logs/shell_logs/`, so old
checkpoints and results are not reused or overwritten. Snapshots contain the full
learner, class statistics, class order and Python/NumPy/PyTorch/CUDA RNG states.
Only load snapshots produced by this script in this checkout and environment.

The comparison fixes ImageNet-R T10, 20 classes/task, seed 1993, rank 64, 20 epochs,
CA5, adaptive private rank, Task0 unmasked + anchor10, Energy50/floor10, global
private conflict and suppress merge. `max_tasks=3` is an actual trainer stop at
Task2 (zero-based); the printed averages are **three-task screening results**.
No automatic ten-task continuation is scheduled.

Both machines can run the same script for seed1993 replication. Compare QV vs
QKV **within each machine**, not the raw accuracy between GPUs. To use another
seed in Fish, set `set -lx SEED 1996` before `lrun`; Bash users can export SEED.

```bash
bash scripts/9_13_qv_after_task0.sh --dry-run
bash scripts/9_13_qv_after_task0.sh --check
```

Look for identical `Task0 checkpoint resumed` SHA256 and Task0 accuracy in the
two resumed logs. QV should report `LoRA projections: QV` at Task1–2 and
`K_safe_norm=0.000000` in every merge log. Assess Total/Old/New, Forgetting and CA
task-prediction metrics together. Confirm gains across seeds before full T10.

## Local validation

Validated on CPU with PyTorch 2.0.1 and timm 0.6.12:

- 17 targeted tests pass, including a real tiny ViT Task0–2 training smoke test,
  K-gradient/merge invariance, checkpoint RNG restoration, and trainer stop bounds.
- Default QKV matches the original `102b65c` attention implementation bit-for-bit
  after a small three-task training/merge comparison; RNG states also match.
- The broader 56-test run has 55 passes and one pre-existing float32 entropy
  tolerance failure: both baseline and candidate return `0.9999998807907104`
  where the old test expects 1 to seven decimal places. No unrelated test or
  training math was changed to hide it.
- Full ImageNet-R training and CUDA validation must be run on the server.
