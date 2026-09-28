# Frozen P-A initialization: 3090 short screen

## Approved scope and hypothesis

ImageNet-R seed1993, Task0–2 only. Test whether a frozen P-A basis aligned to current
training inputs is more useful than the existing random Kaiming basis, without unfreezing A.
This is an initialization experiment, NOT an old-feature subspace penalty or gradient
rectification. No historical data/statistics, extra loss, inference routing or rank redistribution.
Activation principal directions are inspired by EVA, but this is not a reproduction of EVA:
https://arxiv.org/abs/2410.07170 . Energy capture is not evidence of class discrimination.

## Implementation and checks

1. Keep all original A/B allocation calls and their RNG order. After every layer has been
   initialized and trainability set, but before optimizer construction, replace only P-A.
   Task0 and the default `plora_a_init_mode=off` skip both collection and replacement.
2. For Task1 and Task2, sample up to192 current training images (4 batches of48), uniformly
   without replacement using a task-local generator. Use evaluation transforms, NOT test images.
   Hooks observe each attention block's normalized input, including CLS and patch tokens,
   at the same zero-B initial model. Accumulate the uncentered second moment `X.T @ X / N`.
   All layers are collected before any A changes. No extra epochs or training passes.
3. `activation`: CPU float64 eigendecomposition of each input second moment; take the top-r
   eigenvectors as A rows, where r is the existing adaptive P rank.
   `random_orthogonal`: local seeded Gaussian QR yields an independent rank-r orthonormal basis.
   Both retain every corresponding original Kaiming row norm. Neither changes B or the S branch.
   Both run identical feature collection; random control does not use eigenvectors.
4. Restore Python/NumPy/Torch RNG and module train/eval flags; remove hooks even on failure.
   Discard second moments after initialization. LoRA rank, trainable scalar counts and inference
   forward count remain unchanged. Initialization has extra feature collection and decomposition cost.
5. Tests cover formulas/scale, isolated RNG, cleanup, real Attention_LoRA zero-B equivalence,
   frozen A/B training and merge equivalence, lifecycle ordering and complete recipe/CLI matching.
   No validation guards/assertions are added to the training path or launcher.

## Queue and comparator

Original project/branch: `codex/mask-budget-comparison-20260924`, implementation base18d8ea7.
Two runs in order: activation basis, then random-orthogonal control. No baseline rerun.
Anchor2.5, 20epochs, CA5, math-SDPA, original layer dual gates, original scores/losses/rank,
P and S LR multipliers1, A frozen after Task0, full training set, official test reporting.
Other candidate mechanisms explicitly disabled. Machine JSON owns data_path/device.
Weights are saved after each task; these are evaluation weights, not exact resumable training state.

Reference log:
`/Users/luxiaogen/Desktop/loda_logs/9-28/9_28_imgr10_anchor2p5_save_t10_3090.log`.
Matching first-three-task curve:97.10 /92.73 /90.24; T3 Average93.3567;
Task2 Old91.60, New87.29, Forgetting2.155. Full T10 Average87.281 is NOT a T3 comparator.
The reference completed Task0–2 in about22minutes; allow about50–65minutes for the two
candidates plus bounded initialization overhead, assuming similar GPU availability.

## Commands (run from original server project)

Optional GPU smoke, separate from full runs, two tasks and one epoch per task:

```bash
bash scripts/9_28_imgr10_plora_a_init_t3_3090.sh --smoke
```

Formal short screen (two runs, no automatic smoke/test blocking):

```bash
bash scripts/9_28_imgr10_plora_a_init_t3_3090.sh
```

The launcher prints code revision/dirty status, exact CLI and exit codes, uses unique timestamped
logs/checkpoint prefixes and continues to the second candidate if the first fails. `--dry-run`
prints commands without training. The JSON matrix was also expanded with the research skill's
generator for syntax validation; the delivered launcher retains the project's existing optional
smoke, dry-run and timestamp conventions.

## Interpretation and stop rule

Check Task0 matches, `P-A initialization` appears only atTask1/2 for all12layers, B norm0,
A not trainable, ranks/row norms/sample hash matched. Input energy capture alone is not improvement.
Compare Task1 and Task2 Total/Old/New and T3Average/Last/Forgetting with the existing baseline
and with the random-orthogonal control. A gain over random alone but not Kaiming is insufficient.
If New improves at substantial Old cost, or both candidates are flat/worse, do not schedule T10
or expand rank/sample/scale searches. A clear net gain can motivate one fullT10 confirmation;
short test-based screening remains exploratory, not independent generalization evidence.
No extra work is queued on the shared5090. No server training is launched by Codex.

## Verification status

Focused CPU tests:10 passed, including real attention training/merge and full recipe matching.
Compilation, Bash syntax and normal/smoke dry runs passed. Full discovery:303 entries,
301 passed;2 existing modules could not import because local `easydict` is missing
(`test_ca_diagnostics`, `test_task0_margin_screen`). Independent code review approved,
including a separate successful run of all10 focused tests.
GPU smoke/performance not yet verified; supplied commands are for user execution.
