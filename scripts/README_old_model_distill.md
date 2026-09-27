# Previous-model distillation, 3090

One exploratory ImageNet-R T10 seed1993 candidate; no baseline rerun.
Reuse the matching 3090 anchor10 frozen-P-A baseline (head_balance off).
Keep 20 epochs, CA5, math-SDPA, gates, adaptive ranks and learning rates.
Defaults remain unchanged: old_model_distill_weight defaults to zero.

At the start of Task1–9, before update_fc or current LoRA initialization,
copy the previous merged, post-CA model, freeze it, and set eval mode.
Task0 has no teacher (and no CA); its unchanged trained model is Task1's
teacher. Pass the same augmented current training images to teacher and
student. Use teacher.interface for all old classes, not teacher.forward
which returns only its final task head. Student old logits use current
features and detached old classifier weights.

Loss = original loss + weight * T^2 * KL(teacher || student), restricted
to old classes. Both cosine logits are multiplied by the existing scale20
before division by T. Fixed exploratory settings: weight1, temperature2.
No new-class logits enter this KL; no old images or test targets are used.
Release teacher after LoRA training, before class statistics and CA.
Inference stays unchanged. This preserves relative old-class predictions,
not a guarantee of old/new balance, and is not a novel distillation method.

The user chose a fixed full T10 run rather than holdout parameter selection.
Do not describe weight1/T2 as validated or tuned. Loss telemetry includes
old_model_distill and old_model_distill_weighted in existing epoch metrics.
Training adds one no-grad teacher forward and a frozen model copy; GPU
memory and runtime must be measured. There is no server launch from Codex.

Optional independent GPU route check (not automatically inserted):

```bash
bash scripts/9_27_imgr10_old_model_distill_3090.sh --smoke
```

Formal run:

```bash
lrun scripts/9_27_imgr10_old_model_distill_3090.sh ./logs/9_27_imgr10_old_model_distill_3090.log
```

Smoke uses Task0–1 and one epoch, and is not performance evidence. Main
queue has no runtime assertions, automatic tests, or smoke gates. Local
JSON retains the machine data_path. Compare Average/Last, Old/New and
Forgetting; New loss alone is not success. A pure Old-for-New tradeoff
does not establish a useful improvement.
