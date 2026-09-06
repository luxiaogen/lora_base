# DualMask LoRA

本仓库以 DualMask LoRA 作为主实现，不再通过 `ideas/` 下的实验子类加载。

## 代码结构

- `methods/dlora.py`：持续学习训练流程、DualMask 训练损失和任务生命周期。
- `models/attention.py`：LoRA 参数、预训练锚点、保护/冲突掩码以及安全合并。
- `models/network.py`：使用 DualMask attention 的 ViT 和分类网络。
- `utils/dual_mask_metrics.py`：原型能力、漂移和功能合并诊断。
- `exps/dlora/`：CIFAR-100、CUB-200、ImageNet-A 和 ImageNet-R 配置。

`model_name` 的 `dLoRA` 与 `dual_mask_branch` 名称均指向同一份主实现，后者仅用于兼容已有实验记录。

## 运行

```bash
python main.py --config exps/dlora/imgr10.json
```

可使用重复的 `--set KEY=VALUE` 覆盖配置，例如：

```bash
python main.py \
  --config exps/dlora/imgr10.json \
  --set 'seed=[1993]' \
  --set dual_mask_vis=false
```
