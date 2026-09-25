# Mask regularization paired ablation

ImageNet-R T10, seed1993, same-machine sequential pair:

1. `reg001_baseline`: `dual_mask_reg_weight=0.01`.
2. `reg0_candidate`: `dual_mask_reg_weight=0`.

Only the mask penalty weight changes. The candidate removes both protection and
conflict loss terms controlled by this weight, **not** the S/P forward or merge
gates, S protection mask, P plastic mask, or independent Task0 anchor (weight10).
The common recipe matches the preceding P-LR baseline: P multiplier1 / LR0.02,
20epochs, CA5, math-SDPA, layer conflict selection, original adaptive settings.
Data paths are read only from each machine's `exps/dlora/imgr10.json`.

## Run (Bash)

From the existing project, in its training environment:

```bash
# Smoke: Task0 and Task1, one epoch, offline tracker.
bash scripts/9_25_imgr10_mask_reg_3090.sh --set max_tasks=2 --set init_epoch=1 --set epochs=1 --set ca_epochs=1 --set wandb_mode=offline

# Full pair; no smoke overrides.
bash scripts/9_25_imgr10_mask_reg_3090.sh
```

Runs have timestamped prefixes and separate logs under
`logs/shell_logs/imgr10_mask_reg_3090`. The script reports the code revision and
captures a configuration/source-hash fingerprint. Overrides are printed in each
training log. On the recent3090 baseline (~78min/run), allow about2.5–3hours for
the full pair; gradient-diagnostic overhead is not yet GPU-timed.

## Read-only gradient diagnostic

`dual_mask_reg_grad_diagnostic=true` samples the first batch of epochs1/10/20
in Task1–9. It reuses the existing graph: no extra forward, data iterator,
optimizer step, or modification of `.grad`. It logs `MaskRegGrad` records for
S/P separately, aggregating the current trainable B tensors across layers:

- classification gradient norm;
- **weighted** mask-regularizer gradient norm and ratio to classification;
- cosine between these gradients;
- weighted mask loss (both branches together).

The weight0 run reports zero applied regularizer gradient and undefined cosine
(`None`), not a hypothetical gradient at weight0.01. Anchor is not included in
this diagnostic. With zero-initialized B, the baseline's epoch1 first-batch
regularizer gradient can also be zero; epochs10/20 measure the learned update.
Negative cosine means local first-order objective opposition,
not proof that regularization hurts final accuracy or old classes. These are
loss gradients, not momentum/weight-decay-adjusted optimizer directions.

Focused tests compare diagnostic-on/off gradients, parameters and SGD momentum
bitwise over three steps, plus RNG preservation, unchanged gates, retained
Task0 anchor, sampling schedule, and exact script/spec agreement. CPU unit tests
do not replace a CUDA smoke or a completed full experiment.

Compare stage-averaged Old/New over Task1–9 alongside Average, Last and
Forgetting. New improvement alone is insufficient if Old/overall performance
deteriorate. This is a regularization ablation, not a new gradient correction.
