# ImageNet-R QV and conservative P-weight diagnostics

Both sweeps start from scratch under the existing ImageNet-R T=10 protocol: rank 64, 20 epochs per task, CA5, Task0 unmasked with anchor weight 10, and global suppress with Energy-50% plus a Top-10% floor. They read `data_path` from `exps/dlora/imgr10.json` and do not load a Task0 checkpoint.

| Machine | Runs | Only changed variable | Purpose |
| --- | ---: | --- | --- |
| 5090D | 3 | read-only P-conflict diagnostics enabled | Compare ordinary `ones`, old `soft`, and entropy-controlled `conservative` predictions on the same QKV model for seeds 1993/1996/1997. |
| 3090 | 6 | QKV versus QV for all tasks | Matched same-machine comparison for seeds 1993/1996/1997; QV zeros the K rows in both S and P updates during training and merge. |

The conservative rule has no learned router or threshold. For task probabilities `p`, normalized entropy gives `c = 1 - H(p) / log(T)` and the P-conflict weight is `w = 1 - c * (1 - p)`. Flat evidence gives `w = 1`, exactly the ordinary model; sharp evidence moves toward the original soft weights. This remains an inference-only diagnostic.

Interpret QV only against its paired QKV run on the 3090. Interpret conservative only against `ones` and `soft` from the same 5090D model. Do not compare raw 3090 and 5090D scores as a method effect.
