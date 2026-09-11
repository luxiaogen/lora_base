# 梯度隔离对照（ImageNet-R，先到 Task2）

仅新增两个显式模式，默认 `task_local` 不变，不修改 attention、门控、merge、CA 或推理。

| 主训练模式 | 原任务内 CosFace | 新样本全已见类 CosFace | 旧类伪特征 CosFace |
| --- | --- | --- | --- |
| task_local（基线） | LoRA + 当前头 | 无 | 无 |
| task_local_head（3090） | LoRA + 当前头 | detached 特征，仅当前头 | 无 |
| task_local_head_replay（5090） | LoRA + 当前头 | detached 特征，仅当前头 | 仅当前头 |

各项损失系数均为 1；不增加待搜索的权重。旧头在主训练中仍冻结，CA 阶段仍按原实现训练所有已见头。
旧伪特征来自保存的**训练集**均值/协方差（不缩放均值），每旧类 256 个银行样本，每个 batch 抽与新样本同数量的伪特征，使用独立随机数生成器。不使用测试图像或测试特征进行更新。

这不是独立 task router，也不是解决跨任务混淆的保证。它只检验：跨任务竞争不直接更新 LoRA 后，能否避免此前的大幅下降。当前头的改变仍可能通过后续任务内损失间接影响特征训练。

## 运行

先在各机器激活原训练环境，从仓库根目录启动。脚本不切换目录、不修改数据集，也不切换 fused SDPA。

使用现有 `lrun 脚本 总日志` 启动方式时，每台机器调用自己的入口文件：

```fish
mkdir -p logs
# 3090 上执行
lrun scripts/9_11_head_isolation_3090.sh ./logs/9_11_head_isolation_3090.log

# 5090 上执行
lrun scripts/9_11_head_isolation_5090.sh ./logs/9_11_head_isolation_5090.log
```

两个入口都调用 `9_11_head_isolation_pair.sh`，只选择实验模式。
数据路径统一读取各服务器 `exps/dlora/imgr10.json` 中的 `data_path`，脚本不覆盖它，无需再传目录。
可用 `bash scripts/9_11_head_isolation_3090.sh --dry-run` 检查将要执行的命令（5090 同理）。
`lrun` 是服务器已有的自定义命令，本仓库不定义它；直接前台运行仍可使用下面的 Bash 命令。

```bash
# CPU 梯度、检查点、训练循环测试；不加载图像数据。
bash scripts/9_11_head_isolation_pair.sh --smoke

# 3090：本机基线 vs 仅新样本 detached 校准。
bash scripts/9_11_head_isolation_pair.sh 3090

# 5090：本机基线 vs detached 校准 + 旧伪特征。
bash scripts/9_11_head_isolation_pair.sh 5090
```

默认只跑 seed1993；两台机器**各自生成**一个 Task0 checkpoint，然后基线和候选都从该文件恢复，训练 Task1–2。保持原 10-task 类划分与每任务 20 epochs，不把 total_sessions 改成 3。每台机器共 3 次进程启动、5 个任务的训练量。

可先打印完整命令，不训练：`bash scripts/9_11_head_isolation_pair.sh --dry-run 3090`。
可选环境变量：`DEVICE`（默认 0）、`WANDB_MODE`（默认 online）、`PYTHON_BIN`（默认 python）、`SEEDS`（默认 1993，空格分隔）。Fish 多 seed 示例：

```fish
env SEEDS="1993 1996 1997" bash scripts/9_11_head_isolation_pair.sh 3090
```

不要直接把两个 GPU 上的候选分数相减；先计算各机器相对本机基线的变化。要严格比较有无 replay，之后需在同一 GPU 上补相应对照。

## 检查点与结果检查

- 日志分别打印 requested/effective 模式，Task0 有效模式必须是 `task_local`。
- 两个续跑日志的 checkpoint SHA256 和恢复的 Task0 top1 必须相同。
- 检查点保存 learner 全状态（含非注册张量、旧类统计量、曲线和 Python/NumPy/Torch/CUDA 随机状态），不保存 DataLoader。每任务原本重新创建优化器，故无须恢复 Task0 优化器。
- 这是**同代码、同环境、同设备**的实验快照，不是跨版本通用模型格式。只加载自己生成的可信文件：其加载使用 Python pickle。数据签名检查加载后的路径、顺序、标签和划分，**不校验图像文件内容**。
- baseline/candidate 都重新加载同一个快照，避免把序列化差异混入单侧。GPU 非确定性仍可能存在；相同 checkpoint 不等于全程位级确定。
- 日志中的 `PARTIAL DIAGNOSTIC` 表示只完成 Task0–2，Average 不能和完整 T10 的 87.x 比较。WANDB 曲线恢复 Task0 历史，但续跑只上传 Task1–2 的逐任务记录；耗时不含复用的 Task0。
- 比较 Task1/2 的 Old、New、全局准确率、Task Prediction Accuracy、Forgetting，以及 CA 前后混淆。若 Old 和任务识别率仍明显变差，停止，不自动扩展整晚实验；若保住基线，再补完整 T10 和多 seed。

`classification_local` 是原任务内损失；`classification_real` 是新样本全已见类损失；`classification_head` 是新增校准总损失；replay 模式另外记录 `classification_replay`。
