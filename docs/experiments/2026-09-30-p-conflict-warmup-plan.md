# P 冲突门渐进加强：单候选实验

状态：已完成。3090 队列于 2026-09-30 18:50（北京时间）启动，正式 T10 于 20:16 完成。
短测及正式运行均正常；没有净提点，不替换基线。

## 问题与唯一变化

检验在相同最终保护规则下，前期减弱 P 冲突门能否改善新任务学习，而不损害 Old。
不是更换预算、冻结掩码、交替训练 S/P，也不是修改正则损失。

新增开关 `p_conflict_strength_warmup=true`，默认关闭。仅 Task1–9 的 P 训练前向：

| 一基训练轮次 | 本任务原有效 β 的倍率 |
| --- | ---: |
| 1–5 | 0.50 |
| 6–10 | 0.60、0.70、0.80、0.90、1.00 |
| 11–20 | 1.00 |

倍率乘的是包含旧类重叠自适应后的有效 β，不是固定将 β 设置为 0.25。
Task0、S 分支、掩码评分与选择、P 可塑区、所有正则、学习率和 CA 均保持原规则。
评估及 merge 始终使用原 β；后十轮训练前向也已恢复原 β。
不新增稠密状态，不保存 checkpoint，不改本机 JSON 的数据路径。

## 正式协议与参照

3090 原项目，AugReg ViT-B/16，ImageNet-R T10，seed1993。
Task0 及增量任务均 20 epoch；anchor2.5、LoRA LR0.02、CA5、math-SDPA。
layer 粒度、原冲突评分、自适应预算 1×、S/P 双门、正则权重 0.01。
完整显式配置见 [sweep](../../scripts/sweeps/imgr10_p_conflict_warmup_3090.json)。

复用最近同机 layer 基线，不重新训练基线：

`/Users/luxiaogen/Desktop/loda_logs/9-30/9_29_imgr10_granularity_budget_refresh_3090.log`

前缀 `imgr10_granularity_budget_refresh_3090_layer_budget1_seed1993` 的完整 T10 段，
代码 `862f199`：Task0 97.10、Average 87.210、Last 82.45、Old 82.05、New 86.18、Forgetting 5.6867。
不可使用该文件开头一轮短测作为对照；也不混用旧 anchor 或其他机器结果。
默认关闭新开关时的输出、梯度、权重和 RNG 不变由测试检查。

## 执行与记录

```bash
bash scripts/9_30_imgr10_p_conflict_warmup_3090.sh
```

脚本先执行一次 GPU 短测：Task0 1 轮、Task1 11 轮、CA1、W&B offline，
覆盖完整强度变化边界。短测成功后自动启动唯一一个正式 T10 候选；短测分数不作性能证据。
内层日志目录按时间戳和 smoke/run 区分，前缀包含模式与 PID，避免覆盖。
日志中每个增量 epoch 输出 `PConflictWarmup`，包含倍率、原 β、训练 β、评估/合并 β。
实际代码版本由脚本运行时输出；数据及预训练路径继续读取服务器原配置。

## 验证与结果边界

专项测试检查调度边界、真实 epoch 循环接入、Task0/关闭开关不变、S 不变、
正则及选点不变、两分支 B 均更新、A 仍冻结、eval/merge 使用原 β，
以及当前能量/旧重叠自适应规则。所有断言只位于 test，不在训练路径加性能断言。

以阶段平均 Old/New、Average、Last、Forgetting 共同判断。只有 New 改善且整体成绩及
Old 没有明显代价，才支持候选；仅 New 上升、Old 下降仍是取舍，不能称为提点。
前期放松可能损害 Old，后期恢复门强度不保证自动恢复旧知识。
单 seed 结果只作为初步证据。按最近 3090 完整训练约 80 分钟估算，本次另含短测。

## 2026-09-30 部署与启动

- 训练代码提交：`069a0806a3412d83f8996ce2b78d2a94e68143c2`，已推送原实验分支。
- 3090 原项目：`/home/shengqin/lys/baseline/LoDA_ICML2026`，运行于该提交的 detached HEAD；没有新建 worktree 或项目。
- 服务器 GitHub 拉取因 SSH 公钥认证失败；改用包含该提交的 Git bundle 部署，未修改服务器 GitHub 登录配置。
- Python：`/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python`；93 项相关服务器测试通过，Shell 语法检查通过。
- 本地专项/相关测试 85 项通过；完整发现 362 项中两项因本机缺少 `easydict` 导入失败，不冒充完整回归通过。
- 队列 PID：`2861433`；外层日志：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/9_30_imgr10_p_conflict_warmup_3090_20260930_185039.log`。
- 18:51 检查：Python PID `2861450` 使用 GPU，短测 Task0 一轮结束；该分数不作性能结果。
- 18:52 检查：Task1 epoch1–3 输出倍率 0.50，原有效 β=0.5404、训练 β=0.2702、评估/merge β=0.5404；渐进路径实际启用。短测仍未完成，不提前标记正式结果。
- 数据路径、预训练路径继续来自服务器原 JSON；正式配置与最新本机 layer 基线配对，保留原正则、CA5、math-SDPA 和 anchor2.5。

```bash
tail -f logs/9_30_imgr10_p_conflict_warmup_3090_20260930_185039.log
```

## 2026-09-30 完整结果

来源：`/Users/luxiaogen/Desktop/loda_logs/9-30/9_30_imgr10_p_conflict_warmup_3090_20260930_185039.log`。
只比较正式 T10 段；配置除预热开关和运行名称／日志组外与上述基线一致。

| 指标 | layer 基线 | P 预热 | 差值（pp） |
| --- | ---: | ---: | ---: |
| Task0 | 97.10 | 97.10 | 0 |
| Average | 87.210 | 87.151 | −0.059 |
| Last | 82.45 | 82.38 | −0.07 |
| 最终 Old | 82.05 | 81.92 | −0.13 |
| 最终 New | 86.18 | 86.70 | +0.52 |
| Forgetting | 5.6867 | 5.9756 | +0.2889 |

Task1–9阶段平均 Old −0.090pp、New +0.140pp，仍是稳定性／可塑性的取舍。
180条 PConflictWarmup 覆盖9任务×20轮，强度变化确实启用，正式耗时80.04分钟。
单次微小负差不能证明显著退步或确定是波动，但不支持保留本候选。
