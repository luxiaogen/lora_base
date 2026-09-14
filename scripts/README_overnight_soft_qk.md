# ImageNet-R full-T10: P-conflict soft inference and QK-only LoRA

Run from the repository root with the machine's own Python environment. Both scripts read `data_path` from `exps/dlora/imgr10.json`; neither overrides it or resumes a Task0 checkpoint.

| Machine | Script | Ordered runs | Approximate duration |
| --- | --- | --- | --- |
| 5090D | `scripts/9_14_imgr10_soft_qk_5090.sh` | For each seed 1993, 1996, 1997: QKV with same-model P-conflict diagnostics, then QK-only | 6–9 h |
| 3090 | `scripts/9_14_imgr10_qk_all_3090.sh` | QK-only for seeds 1993, 1996, 1997 | About 8 h |

Every run is a fresh ImageNet-R 10-task/20-class-per-task training with rank 64, 20 epochs per task, CA5, Task0 unmasked, Task0 W0 anchor weight 10, global suppress, and Energy50/Top10 floor. In QK-only runs, both S and P LoRA have zero V-row gradients and V rows receive no LoRA merge update at **every task, Task0–9**. Full QKV storage remains allocated, so this is a projection ablation rather than a parameter-count reduction.

The 5090D QKV run still trains and performs CA exactly as before. Its `P-conflict diagnostic` logs compare ones (ordinary CNN baseline), uniform, input-derived soft weights, and an oracle using true task ID. Only the first three are label-free; oracle is diagnostic, never a class-incremental score. Soft weighting is **inference-only** in this sweep, not a training-time method. The 5090D QK paired run enables the same read-only diagnostics, so the QKV/QK training comparison changes only the QK switch. The 3090 QK runs omit the extra diagnostics to fit the night; they are cross-machine replication, not the primary matched comparator.

Compare QKV versus QK by seed **within the 5090D** first; check Task0 accuracy, final Average/Last, Old/New, Forgetting and task prediction. Compare soft versus ones and uniform **within each QKV run**. Do not treat a 3090–5090D accuracy difference as a method effect. Runtime references: prior full-T10 QKV runs took about 48 min each on 5090D and 2.6–2.7 h each on 3090; dense component diagnostics add inference and storage cost.
