# Idea 2: Dual-Mask LoRA Branch

This branch implements the dual-mask direction as an independent experiment.
It does not use full Fisher information for the pretrained weight `W0`
importance by default.

- `general_mask` marks important pretrained `qkv.weight` entries to protect.
- `isolated_mask` is the complementary plastic region.
- `w0_importance` is estimated from SVD, gradient sensitivity, or both.
- `BA` importance is estimated from the current LoRA update magnitude.
- `normalize(I_W0) * normalize(I_BA)` approximates W0-BA conflict.
- High-conflict BA updates are suppressed during forward and final merge.
- A lightweight conflict regularizer can be enabled with
  `dual_mask_reg_weight`.

Default config uses `svd`, which avoids an additional backward probe. To include
gradient sensitivity, set:

```json
"dual_mask_importance": "svd_grad",
"dual_mask_grad_batches": 1,
"dual_mask_grad_alpha": 0.5,
"dual_mask_reg_weight": 0.05
```

Run:

```bash
python main.py --config ideas/dual_mask_branch/configs/cifar10.json
```
