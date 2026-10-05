# DualMask 纯基线（debug 分支）

分支：`xiaogen/dualmask-baseline-clean`。只保留原 DualMask 的训练与推理实现，不把候选功能设为关闭后留在代码里。

来源提交：`e72bc41a0e6af502b46c62bc3e6242b28b851e8e`，对应最近 A0 的原非对称权限配方。清理不是新算法，也不代表重新验证了完整 ImageNet-R 成绩。原开发分支及 Git 历史仍保留实验代码、日志来源和台账。

## 保留的算法

- 固定的 AugReg ViT-B/16 预训练 QKV 权重 W_pre；预训练文件和原分支相同。
- 用 W_pre 的 SVD 结构重要性构造保护区，P 可塑区为其补集。
- 通过当前训练数据的预训练 NCM 能力、需求和旧类竞争量，调节保护、冲突强度及 P rank。
- Task0：只使用 S 分支，A/B 均训练，unmasked；独立 anchor 权重为 2.5。
- Task1 起：S/P 的 A 随机初始化后冻结，只训练 B。S 软保护，P 硬补集限制，均使用原乘积冲突门。
- 保留原掩码正则；门控参与训练前向和任务结束合并，合并后移除临时适配器。
- 保留原 CA（类别均值、协方差及伪特征），推理为单个当前骨干、全部已见类别分类，不输入真实任务 ID。

不包含 Oracle、教师/蒸馏、Ridge/NCM 融合推理、白化/梯度/方向初始化、权限候选、位置打乱、CA 候选、额外诊断、夜间队列或 checkpoint 保存。控制器中必要的 NCM 计算不是额外推理专家。

## 代码入口

| 文件 | 适合设置断点的位置 |
|---|---|
| `main.py` | 读取配置与命令行覆盖 |
| `trainer.py` | 随机种子、math-SDPA、任务循环及正式指标 |
| `methods/dlora.py` | `_prepare_w0_prototypes`、`_train`、`_extra_training_loss`、`_stage2_compact_classifier` |
| `models/attention.py` | `before_task`、`_svd_importance`、`rebuild_dual_masks`、`_joint_conflict`、`_safe_delta`、`after_task` |
| `models/network.py` | 当前任务训练 head 和全类别 `interface` |
| `models/vit.py` | ViT 骨干及原 AugReg NPZ 权重加载 |
| `exps/dlora/imgr10.json` | 唯一工作配置及可修改的数值超参数 |

## 运行

使用原服务器的 CUDA 环境，不要为这个分支重新安装/降级 torch 或 torchvision。完整预训练模型加载要求原代码使用的 `timm==0.6.12`；timm 1.x 的配置对象与原 builder 不兼容。本次未升级服务器依赖。

数据目录需要已有 `train/<class>/*.jpg` 和 `test/<class>/*.jpg`。不自动移动、删除或重新划分数据。

在仓库目录运行（把路径替换为本机已有 ImageNet-R 目录）：

```sh
python main.py --config exps/dlora/imgr10.json \
  --set data_path=/path/to/imagenet-r
```

默认：seed1993、20 轮、CA5、anchor2.5、完整 T10、math-SDPA，不保存权重。数值设置可通过现有配置键覆盖；已经删除的实验选项会在命令行入口报错，不会静默忽略。

仅调试 Task0–1、各一轮（不是正式成绩）：

```sh
python main.py --config exps/dlora/imgr10.json \
  --set data_path=/path/to/imagenet-r \
  --set max_tasks=2 --set init_epoch=1 --set epochs=1 \
  --set ca_epochs=1 --set prefix=debug_two_tasks
```

日志：`logs/ImageNet_R/dualmask_baseline/<prefix>/1993.log`。

## 清理一致性与测试

```sh
python -m unittest discover -s test -v
```

测试从固定的来源提交读取旧实现，只在测试进程中对照；训练入口不会读取旧源码。请使用普通 Git clone，浅克隆需要先补齐来源提交历史。

14 项测试通过：已核对初始化、随机状态、保护/冲突门、正则及梯度、前向与合并、W_pre 恢复和预训练来源配置。三任务 CPU 小模型对照包含真实训练循环、类别统计与 CA；对应参数、统计、指标和 CPU 随机状态逐项一致。该测试直接构造小模型，不下载预训练权重，不是完整 ViT 的初始化、CUDA 或 T10 性能复现。

为了删掉旧诊断又保留随机序列，有两处很小的兼容处理：

- `utils/reproducibility.py` 替代旧的额外 DataLoader 迭代产生的随机 seed 消耗，不再执行那些诊断前向。
- `models/vit.py::init_weights` 只临时产生旧 growing-token 初始化的随机数，不保留 growing-token 参数或其他前向。

这些处理不参与损失或选点，保留是为了避免“只删无用代码却换了下一任务初始化”。
