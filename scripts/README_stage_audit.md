# Three-stage report, ImageNet-R T10 seed1993

3090: anchor10 baseline, one full run. 5090: anchor10 then anchor5, two full runs.
Both keep 20 epochs, CA5, math SDPA, original layer dual gates, P LR .02,
mask regularization .01. No P direction correction. Dataset paths remain in local JSON.

Each script first runs Task0–1 with one epoch/CA1, checks all three reports,
then starts full training. Per-run logs and transition CSV are timestamped under
`logs/shell_logs/imgr10_stage_audit_<gpu>/`. Run with bash or the user's lrun.

The opt-in `stage_audit=true` records:

1. pre_merge: training finished, before after_task merges/discards adapters;
2. post_merge: immediately after all adapters are merged;
3. post_ca: after class-statistic extraction and CA (Task0 has no CA).

All three use the same sequential test loader and network.interface, globally
classifying all seen classes. Metrics: Total/Old/New accuracy, raw cosine-logit
margin/CE, corrected/broken counts and maximum logit change relative to the
previous stage. Sample-index/label hashes and class counts are checked by the
launcher. No images or historical feature covariance are retained. CPU logits
are held only until the next stage, and discarded at the end of the task.

Python/NumPy/Torch/CUDA RNG and module train/eval flags are restored. No gradients,
weights, hyperparameters or predictions are selected using these reports.

Important: these are TEST REPORTS, not a train holdout or validation experiment.
Training data are unchanged to preserve the baseline. Do not select optimal
epochs/strengths from these reports or claim validation generalization. A later
method-selection experiment needs an independent training holdout. Anchor5 is
a predeclared follow-up, not an automatically selected configuration.

Post_merge -> post_ca also spans class-statistic extraction; current extraction
does not optimize weights. If the backbone pipeline later changes, revisit this
attribution. Global New accuracy measures old/new competition, not just the
within-new-class training accuracy. Merge equality is measured, not assumed.

No speed or performance benefit is claimed. Local unit tests do not replace the
server's GPU smoke. Plan roughly 1.5–2.5h on 3090 (one run), 2–7h on 5090 (two),
given the large runtime variability in the previous night; estimates are not caps.
