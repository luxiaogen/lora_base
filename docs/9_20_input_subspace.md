# P conflict input-subspace screen (5090)

Branch is isolated from main. Three fresh ImageNet-R runs, seed1993, original
20-class increments / ten-task partition, STOP after Task2 (`max_tasks=3`).
20 epochs, CA5, QKV rank64, fresh A/B initialization, existing masks unchanged.
No Task0 checkpoint and no test-data fitting.

Modes: baseline / old / random. Every arm collects identical statistics after
each task's merge and CA: deterministic current TRAIN view, eight images/class,
16 evenly spaced token positions at each attention input. No old images stored.
Accumulate equal-weight task second moments (uncentered), keep 32 eigenvectors.
Old moment snapshots can become stale as the backbone evolves; this is a limitation.
Random control uses 32 orthonormal directions from a local seed per layer and
does not consume the training RNG. Rank is unrelated to LoRA rank64.
Equal rank/coefficient does NOT imply equal penalty or gradient magnitude; compare
logged loss scales before attributing any difference solely to semantic directions.

Task1 onward: add `1.0 * mean_layer ||C_P U||_F^2`; C_P is gamma times the
masked P update restricted to its current conflict mask, exactly as in the
training forward. No gradients into basis or mask. No S penalty, routing,
permanent deletion or changes to merge/CA. Log raw/weighted `p_input_loss` and
captured energy. Weight1 is a fixed pilot choice, not a demonstrated optimum.
Baseline still collects statistics but adds no loss.

Success: old basis beats BOTH baseline and random, without sacrificing newly
learned accuracy. Reduced forgetting alone is insufficient. If loss is negligible,
the result only tests this coefficient, not all subspace methods. Do not auto-select
hyperparameters or advance to T10 based on test labels. Review the three runs first.

`lrun scripts/9_20_p_input_subspace_5090.sh ./logs/9_20_p_input_subspace_5090.log`

These lightweight bases are transient training state; this runner always starts
fresh and does not implement resume/checkpoint serialization of statistics.
