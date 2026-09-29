# Coverage and matched-removal figures

## Figure 1 — completed historical 3090 data only

`plot_conflict_coverage.py` is bound to the fixed-coverage 3090 log from 2026-09-27.
The selected fractions are 0, .1, .4, .6 of full QKV coordinates; energy adaptation
was disabled in those historical runs. The ordinate is the equal-weight mean over
9 tasks × 12 layers of `||removed P update|| / ||P update after plastic gating||`.
It is not weight sparsity or the fractional reduction of the retained norm.
The dashed 50% line is a theoretical reference, not a 3090 measurement at 100%.
No 5090 data are included. Raw logs round ratios to four decimal places.

## Figure 2 — one new 3090 full T10 control

Baseline: `logs/9_28_imgr10_anchor2p5_save_t10_3090.log`, SHA256
`90988385d6baa842b813859338dc55036189837f40478ed1946cd4a767bfb568`.
Reuse this completed selective baseline; do not rerun it.

Candidate: anchor2.5, ImageNet-R T10, seed1993, 20 epochs, CA5, math-SDPA;
original adaptive layer conflict recipe, not the historical fixed-coverage recipe.
Only `dual_mask_uniform_norm_matched=true` changes the training method.
Checkpoint saving is disabled. Local dataset paths remain in the server JSON.

For each current S/P layer update U after S protection or P plastic gating,
calculate the original selective mask M and beta, then use
`alpha = beta * ||U * M|| / ||U||`, detached from autograd, and `(1-alpha)*U`.
Forward and suppress-merge share the same implementation. Task0 is unmasked.
This matches the *counterfactual selective removal on that same current update*.
It does NOT match final retained norms or force two training trajectories to agree.
It replaces spatial selection in both forward and backward; it is not solely a
post-training merge ablation. Existing regularizers and CA remain enabled.

`MatchedConflictNorm` records reference/actual removed norms and base norm for all
216 incremental task/layer/branch combinations. The figure script checks coverage
and relative matching error before plotting completed T10 Old/New values.

Run CUDA smoke first, then the full candidate with automatic plotting:

```bash
bash scripts/9_29_imgr10_uniform_norm_3090.sh --smoke
bash scripts/9_29_imgr10_uniform_norm_plot_3090.sh
```

Figure exports are PNG/PDF/SVG plus source JSON/CSV. Local SciPlot installation
lacks its runner and reference files, so status is `render_only`, not certified
publication-ready. Figure 2 is not a result until the full run completes.
