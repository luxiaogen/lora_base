# Task0 后只训练 P：短程配对验证

基线提交：`102b65c`。实验分支：`codex/p-only-after-task0`。不向 main 合并性能尚未验证的修改。

## 唯一实验变量

`dual_mask_s_task0_only=false`：原始 S+P；`true`：Task0 原样训练 S，Task1 起只训练 P 和当前分类头。

- Task0 仍是 unmasked + W_pre anchor weight=10；后续不加 anchor。
- Task1 起 S 的 A/B 冻结，并从前向、anchor 增量和 merge 中排除；P 保留原保护/冲突规则与自适应 rank。
- S 仍按原顺序初始化（B=0），保持随机数消耗和 P 初始化一致。暂未减少分配显存。
- 保留原正则的 S/P 平均分母，冻结 S 的零项不产生梯度，避免把 P 正则强度顺带加倍。
- P 更新仍合并进累计 QKV，因此不是“冻结旧模型”，也不保证零遗忘。
- 每层记录 `S_active / S_trainable / P_trainable / P_rank`。可训练容量减少是本消融的一部分，不应将结果全部归因于冲突减少。

## 执行

从仓库根目录、训练 Conda 环境运行。数据目录只读取本机 `exps/dlora/imgr10.json` 的 `data_path`；不覆盖配置，不移动图片。

```bash
# 可选预检：CPU 单元测试与 tiny ViT 合成数据训练，不加载 ImageNet-R 或预训练权重
bash scripts/9_12_p_only_after_task0.sh --check

# 打印完整命令，不训练
bash scripts/9_12_p_only_after_task0.sh --dry-run

# 正式短程筛选；3090/5090 都使用同一脚本，各自在本机做配对
lrun scripts/9_12_p_only_after_task0.sh ./logs/9_12_p_only_after_task0.log
```

固定 ImageNet-R 的 T=10 类别划分，但只运行到 Task2；seed1993、S rank64、原自适应 P rank、20 epochs、CA5、Energy50/floor10、global+suppress、reg_weight=.01、conflict_reg=false。不改变 fused SDPA 协议。

执行顺序：

1. Task0 一次，评估并保存完整 learner、旧类统计量、曲线及 Python/NumPy/Torch/CUDA RNG。
2. 从该文件恢复，S+P 跑 Task1–2。
3. 从同一文件重新恢复，仅 P 跑 Task1–2。

共 3 个进程、5 个任务训练量。每次启动建立独立日志目录，任一步失败即停止。两组恢复日志应有相同 SHA256、Task0 曲线；不同机器各自产生自己的 checkpoint，不跨硬件搬运。只加载脚本自己生成的 checkpoint（包含 pickle），不要加载不可信文件。

## 看什么

先确认两个分支都完成 Task2，然后在同一机器内比较 Average/Last、Old/New、Forgetting、Task Prediction 及 CA 前后指标。这里的 Average 只有前三任务，不能与完整十任务的 87.x 直接比较。

优先看遗忘下降是否同时保住 Old、New 和 Last；如果只有 Forgetting 下降而 New/Last 明显变差，应判为容量不足/可塑性损失的证据，不算成功。单 seed 短程结果只用于筛选，通过后再在新实验分支扩展多 seed 和完整 T10。

脚本采用共享 checkpoint 的串行配对执行器，而非独立运行的通用 sweep，避免一次 Task0 失败后误用旧 checkpoint。
