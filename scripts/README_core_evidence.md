# DualMask 双机核心归因夜间队列（2026-10-04）

这是原方法的归因与精简实验，不增加教师、新 CA 或方向初始化，不承诺提点。
统一 ImageNet-R T10、seed1993、anchor2.5、20 epochs、CA5、math-SDPA；不保存权重。
两台各运行一次同提交基线；只作本机配对。

## 运行

在各自原项目及原 Python 环境中运行 Bash 包装：

```bash
bash scripts/10_04_imgr10_core_evidence_3090.sh
bash scripts/10_04_imgr10_core_evidence_5090.sh
```

每台先对所有预排配置各跑 Task0–1、一轮训练和一轮 CA，全部短测通过才进入正式队列。
短测不用于筛选准确率。自动记录 queue PID、实际 Python 子进程 PID 和日志。
自首个短测启动计时；10 小时后不启动任何新的短测/正式组，已启动组完整结束。
运行异常立即保留输出并暂停该机；不重试，不跳过失败继续跑。
该行为按用户要求实现，区别于通用 sweep 生成器的失败后继续行为。

只展开命令、不启动 GPU：

```bash
python scripts/run_core_evidence_night.py --machine 3090 --mode dry-run
python scripts/run_core_evidence_night.py --machine 5090 --mode dry-run
```

## 固定矩阵

| 3090 | 保护位置 | 权限 | 排名 | 位置范数控制 |
|---|---|---|---|---|
| R0 | W_pre | 原非对称 | 乘积 | off |
| R1 | W_pre | 原非对称 | 乘积 | paired_min |
| R2 | 固定置换 | 原非对称 | 乘积 | paired_min |
| R3 | W_pre | 两分支软保护 | 乘积 | off |
| R4 | W_pre | 两分支硬补集 | 乘积 | off |
| R5 | W_pre | 原非对称 | 幅度 | paired_min |
| R6 | 固定置换 | 原非对称 | 幅度 | paired_min |

| 5090 | 消融 |
|---|---|
| S0 | 原基线 |
| S1 | 仅关闭冲突惩罚，W_pre 保护正则保留 |
| S2 | 仅关闭 S/P 冲突门，正则保留 |
| S3 | 冲突惩罚与 S/P 冲突门均关闭 |
| S4 | 全部掩码正则关闭，独立 Task0 anchor 保留 |
| S5 | 原有额外移除范数匹配的均匀冲突抑制 |
| S6 | 仅保留 S 掩码正则，P 项置零、平均分母不变 |
| S7 | 仅保留 P 掩码正则，S 项置零、平均分母不变 |

完整公共配方沿用 `imgr10_protect_position_3090.json` 的 common_overrides；
本轮差异由 `imgr10_core_evidence_night.json` 明确列出，CLI 覆盖本机训练设置，保留本机数据路径。
R3/R4 只比较权限策略，不宣称相同实际更新量。

## 位置范数控制与权限

对同一 BA，计算原位置、固定置换位置的完整有效门（权限×冲突）。
每层、每分支、每 Q/K/V 块，选择两种有效更新范数的较小值；只缩小较大者，系数 detach。
两者均零时系数为1，保留 B 零初始化时的学习梯度；P 原硬补集的零坐标不被放开。
行/列置换在每层 Q/K/V 独立，用私有 generator；跨任务固定，不消耗训练 RNG。

前向、掩码正则和最终合并用同一个未乘 gamma 的 BA 构造门与范数系数。
前向/合并随后应用原 S/P gamma。掩码正则保持原有未乘 gamma 的 BA 能量单位和系数，
不额外乘 gamma²，以免引入第二个损失强度变量。默认 off/asymmetric 保留原路径。
Task0 全部保持 unmasked、独立 anchor，不启用新权限或范数控制。

这控制同一状态的两种位置更新，不保证不同训练轨迹的绝对更新范数相同。
S5 匹配的是冲突额外移除范数，与位置 paired_min 不同。

## 只读机制与成本记录

每 epoch 记录原始、权限后、最终有效及移除更新范数、覆盖率；合并时保留实际交付更新记录。
paired_min 的权限后移除量包含冲突抑制及范数控制，字段使用 `post_permission_removed_norm`；
其 `conflict_removed_norm` 留空，不能把范数控制缩小误归因于冲突门。
Task1/5/9、epoch1/5/10/20：按固定类别轮转顺序最多512旧类、128新类测试样本，
同一输入、同一全部已见类别范围，对比原/置换位置的同状态 paired_min margin/预测。
不将真实标签用于训练、调参或决定继续跑哪组；诊断后恢复模式、位置 override 与 Python/NumPy/Torch RNG。
不保存旧图片、逐样本历史特征或额外大矩阵缓存。

成本记录含：保护/适配器构造、W_pre 能力评估、训练、CA 统计、CA、推理、诊断、更新遥测。
训练耗时包含原训练中已有评估/数据加载，但减去新只读诊断和更新遥测时间。
推理成本包含原有 W_pre-NCM 报告，不可直接写成纯学生一次推理的延迟。
峰值显存为自该任务训练开始 reset 后累计峰值，不是每个子阶段独立峰值；
另记录进程累计峰值 RSS、活跃参数/buffer/CA 均值/协方差/原型字节数，不含优化器与临时工作区。
审计本身增加两个保护掩码缓存；它们属于测量开销，不声称是原方法必要存储。

## 输出与次日判定

`logs/shell_logs/imgr10_core_evidence_<machine>/<timestamp>/` 下：

- `manifest.json`、`active.json`、`smoke_queue.json`、`queue.json`：队列身份与状态。
- 每组 `run.json`、`training.log`：完整提交、源码 SHA256、有效配置、硬件软件、原始日志。
- `results.csv/json`、`tasks/epochs/updates/masks/diagnostics/costs/storage.csv`：真实结果及遥测。
- `report.md`、`contrasts.json`：仅完整 T10 且配置/源码匹配时解释归因。
- `old_new.png/pdf`、`task_curves.png`、`norm_trajectories.png`、`position_margin.png`、
  `stage_costs.png`、`storage_costs.png`：两台分别绘图，不混用绝对分数。
- 完整权限/2×2配对后另生成 `permissions.png`、`gates_penalty_2x2.png`。

队列每完成一组更新汇总。可用原 Python 重新生成：

```bash
python scripts/analyze_core_evidence.py logs/shell_logs/imgr10_core_evidence_3090/<timestamp>
```

时间预算未启动组是 pending，不能作负结果；完整配对不足时不形成相应归因。
遥测完整性与正式性能有效性分别标记；小差异不宣称统计等效/显著。
位置优势要结合两种排名、同状态等范数 margin 和跨轨迹实际范数一起解释。
非对称权限优于两个对称对照才支持权限组织，不证明双分支本身新颖。
单 seed 删除组件的微小差异只能列为精简候选，不自动替换基线。
