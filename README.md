以下是中文翻译，保留了原文结构和代码块，术语采用业界常见译法：

---

# Idea 2：双掩码 LoRA 分支

该分支将双掩码方向作为一个独立实验实现。  
默认情况下，它**不**使用预训练权重 `W0` 的完整 Fisher 信息来衡量重要性。

- `general_mask` 标记预训练 `qkv.weight` 中需要保护的重要条目。
- `isolated_mask` 是与之互补的可塑性区域。
- `w0_importance` 通过 SVD、梯度敏感性或两者结合来估计。
- `BA` 的重要性通过当前 LoRA 更新幅度来估计。
- `normalize(I_W0) * normalize(I_BA)` 用于近似 W0-BA 之间的冲突。
- 高冲突的 BA 更新在前向传播和最终合并时会被抑制。
- 可通过 `dual_mask_reg_weight` 启用轻量级冲突正则化项。

默认配置使用 `svd`，这避免了额外的反向传播探针。如需包含梯度敏感性，请设置：

```json
"dual_mask_importance": "svd_grad",
"dual_mask_grad_batches": 1,
"dual_mask_grad_alpha": 0.5,
"dual_mask_reg_weight": 0.05
```

运行：

```bash
python main.py --config ideas/dual_mask_branch/configs/cifar10.json
```

---
