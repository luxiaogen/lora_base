# ImageNet-R 跨任务竞争验证

目的：单独验证主训练目标，不同时更换 P rank、门控、Anchor 或 CA。

## 三个模式

`classification_training_mode` 默认 `task_local`。Task0 在所有模式下都强制走原任务内训练。

| 模式 | 主训练 |
| --- | --- |
| A `task_local` | 原任务内 CosFace |
| B `all_seen` | 真实新样本使用全部已见类别 logits，标签加回旧类偏移 |
| C `all_seen_replay` | B 的损失 + 等量旧类伪特征的全类别 CosFace，系数固定为 1 |

B/C 均冻结旧分类头；真实样本的梯度通过所有分类头传回当前 S/P 和当前分类头。
C 的伪特征仅更新当前分类头，不更新主干或旧分类头。CA 阶段仍按原方式解冻全部已见分类头。

复用已有训练集均值和全协方差，不读旧类原图或测试特征。每个增量任务开始时，每个旧类采样 256 个伪特征，形成该任务固定采样池；均值不额外缩放，协方差沿用已有的 1e-3 I 稳定项。每个 batch 从池中均匀有放回抽取与真实样本等量的伪特征（旧类在期望上均衡）。使用独立、由 seed 和 task 编号确定的随机数生成器，不推进训练/CA 的全局随机数流。Task9 采样池约 135 MiB，不计临时张量。

C 不会将真实损失减半：`L = L_real + L_replay + 原有正则`。因此它是增加旧类约束的消融，不是与 B 完全相同的分类头梯度尺度。旧统计量仍可能过时，本实验不做映射校正，也不宣称伪特征能保护旧图像特征。

## 固定协议

ImageNet-R，10 tasks，rank64，自适应 P rank；20/20 epochs；CA5；Task0 unmasked + anchor10；global + suppress + Energy50/floor10；conflict_reg=false、reg_weight=0.01；关闭 selective_anchor、safe_residual、functional_merge_calibration。

- 3090：A1993、C1993、A1996、C1996、A1997、C1997。
- 5090：同样 6 次配对，最后 B1993、B1996、B1997。
- 两机各自做同 seed 配对，不跨硬件直接相减。预计原速度约 7.7/7.2 小时；C 的额外开销尚未经 GPU 实测。
- `sweeps/*.json` 是生成来源；5090 合并 main 和 ablation 两部分。脚本使用技能生成器生成，再去掉 cd、加入预检和参数转发。

## 运行（仓库根目录，先激活训练环境）

```bash
# CPU 单元及小型合成训练检查，不加载预训练权重/数据集，不等于 GPU smoke
bash scripts/9_10_cross_task_3090.sh --smoke

# 各机器只选择对应的一条
bash scripts/9_10_cross_task_3090.sh
bash scripts/9_10_cross_task_5090.sh
```

脚本没有 cd。默认在现有 python 环境运行；先执行预检测试，失败立即退出。正式训练某次失败会记录 FAIL，继续余下实验，最后以非零状态结束。日志目录分别为 `logs/shell_logs/imgr10_cross_task_3090` 和 `logs/shell_logs/imgr10_cross_task_5090`，W&B group 也分开。两台机器应确认相同 commit、无实验代码差异、数据集与预训练权重相同。

本机数据路径不同可仅覆盖路径，例如：

```bash
bash scripts/9_10_cross_task_5090.sh --set data_path=/your/imagenet-r
```

其他尾部 `--set` 也会转发，但不要覆盖模式、seed、epoch 等实验变量。脚本按当前环境选 GPU0；可通过 CUDA_VISIBLE_DEVICES 选择物理卡。

## 必看日志

- `Classification training: requested=... effective=...`：Task0 必须 effective=task_local。
- `classification_real`、C 的 `classification_replay`：实际损失；`Train_accy` 仍是任务内准确率，新增 `train_global_acc` 才是主训练全类别准确率。
- `CA diagnostic`：CA 前后 Total/Old/New；新增 `CA task prediction`：CA 前后任务识别率。
- Average、Last、Old、New、Forgetting、Oracle 曲线与 Task Prediction Accuracy 曲线。

诊断不参与训练、采样或 checkpoint 选择。仅 Forgetting 下降不足以证明改善：需要 Old/Last 同时改善，并检查是否只是任务初学准确率降低。

## 验证边界

CPU 测试覆盖真实 MANet 分类路径、真实 Learner 训练循环、旧头冻结、真实/伪特征梯度、全局标签偏移、Task0 与旧循环逐参数及 RNG 一致、采样独立性和 CA 诊断无训练扰动。没有在本地完成 ImageNet-R/CUDA 训练，不保证 GPU 吞吐或性能增益。
