# Transfer continuation (3090, seed1993)

Two full ImageNet-R T10 runs, each FROM SCRATCH after its GPU smoke succeeds:

1. CL-LoRA + DualMask, shared protection 0.25 (previously 0.5).
   Only the shared protected-coordinate multiplier changes from 0.5 to 0.75.
   Private plastic mask, conflict strength0.5, selection, rank, native classifier,
   epochs, and Task0 are unchanged. This is NOT weakening every DualMask gate.
   Compare with completed off / on0.5 runs only after checking fingerprints,
   pretrained checksum, dataset manifest and Task0 consistency.
2. SD-LoRA off baseline, no DualMask. Previous off failed at Task5.
   Non-reentrant activation checkpointing on train-time LoRA QKV recomputes its
   forward on backward, keeping official equations, batch size, magnitudes and
   direction factors intact. RNG is preserved. No old factors detached beyond
   upstream freezing. CPU Task5 test checks outputs and all gradients against
   official forward. CUDA memory savings and runtime still require server testing.

CL smoke: first two tasks, one epoch. SD smoke: first SIX tasks, one epoch,
unchanged batch size, reaching the prior OOM stage. Smoke states are not reused.
Passing Task5 smoke does not guarantee Task9 fits. No automatic batch reduction
or partial result treated as a complete ten-task baseline.

Official revisions stay pinned; launcher ignores only tracked __pycache__/*.pyc
changes and disables bytecode writes. Actual source changes still block execution.
Data comes only from this machine's exps/dlora/imgr10.json; do not copy data paths.

Run: `lrun scripts/9_20_transfer_balance_3090.sh ./logs/9_20_transfer_balance_3090.log`
Smoke only: `env SMOKE_ONLY=1 bash scripts/9_20_transfer_balance_3090.sh`

Prior CL full runtime was ~91min with DualMask. Budget ~3–5h including SD
recomputation and smokes, not a measured guarantee. No retraining of completed
CL off/on0.5 or SD on in this queue. No changes to main.
