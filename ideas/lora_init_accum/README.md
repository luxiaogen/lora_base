# Idea 1: FRLoRA-Style LoRA Initialization and Residual Accumulation

This branch adapts the initialization and accumulation mechanism from
Federated Residual Low-Rank Adaptation (FRLoRA) to the current continual
learning baseline.

- On first use, each frozen `qkv.weight` is decomposed once with SVD.
- The top singular components initialize `B0` and `A0` as
  `B0 = U[:, :r] * sqrt(S[:r])` and `A0 = sqrt(S[:r]) * Vh[:r, :]`.
- The frozen weight is shifted to `W_hat = W - B0 @ A0`, so the effective
  weight `W_hat + B0 @ A0` is initially identical to the pretrained weight.
- Each task starts from the same principal-space `B0, A0`.
- After each task, the learned residual `B @ A - B0 @ A0` is accumulated into
  `qkv.weight`, and the local LoRA branch is reset to `B0, A0`.
- The accumulated residual is stored in `cumulative_lora_delta`.
- The original `dLoRA` model name still uses the untouched baseline modules.

Run with one of the configs under `ideas/lora_init_accum/configs/`, for example:

```bash
python main.py --config ideas/lora_init_accum/configs/cifar10.json
```
