# Anchor5 + old-class competition screen

Both GPUs independently run baseline (weight0) then candidate (weight0.1),
ImageNet-R T10 seed1993, 20 epochs, CA5, math SDPA, anchor5 Task0 only.
Layer dual gates, P LR .02, mask regularization .01 and inference stay unchanged.
Only imgr10.json's default anchor changed to5; other datasets and paths are untouched.

At Task1+, reuse the current forward's features and current-head cosine logits.
Compute old-head cosine logits using detached old classifier weights:

`extra_loss = 0.1 * scale * mean(relu(max_old_cosine - correct_new_cosine))`

Scale is the existing CosFace scale20. Extra margin is zero. Original current-head
classification loss is retained. Gradients reach features and the current head,
not old head weights. This affects trainable S/P branches, NOT P alone. No second
backbone forward, new network, replay data, test labels, or task-ID inference.
Task0 and weight0 skip this computation entirely. CA may still train all heads,
exactly as the original baseline does; freezing here refers to the LoRA stage.

This is a PREDECLARED exploratory comparison, not a holdout-tuned method. Weight
0.1 is fixed before results, not chosen from test curves. Do not automatically
adopt the highest test result or sweep strengths from these test reports. A later
hyperparameter-selection study must use an independent training holdout. Running
the candidate now tests the proposed hypothesis; it does not presume cross-task
errors dominate. If baseline errors contradict that premise, stop this direction.

StageAudit adds counts at pre_merge/post_merge/post_ca:

- new.cross_partition_errors = new predicted old;
- old.cross_partition_errors = old predicted new;
- within_partition_errors = wrong class inside the true old/new partition;
- partition_oracle_accuracy/recovered = diagnosis after restricting candidates
  to the true old/new partition. This uses labels ONLY for reporting and is NOT
  the deployed predictor or ordinary CIL accuracy. Old partition includes all old
  tasks, so it is not an exact task-ID oracle.

Counts use the same global predictions; no extra forward is needed beyond the
existing stage audit. Divide counts by the partition's n for comparable rates.
The CSV exporter includes all added metrics. Old/new error counts must sum to
that partition's total errors. Mean stage New and final New must be distinguished.

Run the GPU-specific 9_26_imgr10_old_competition script with bash or lrun.
Scripts run unit checks, then baseline/candidate 1-epoch Task0–1 smoke before T10.
Smoke checks all three stage reports, matching Task0, and nonzero candidate loss.
Server GPU smoke is pending; local tests are not training-performance evidence.
Machine JSON supplies data_path. Same-machine pairs only, no cross-GPU subtraction.

Decision: report Average/Last/Forgetting, stage mean and final Old/New, new->old
and old->new rates. Reduced new->old errors alone are not success if Old declines.
Retain no claim of improvement until matched full runs show a net benefit. No
automatic selection, new seeds, or coefficient sweep is included in this queue.
