# 保护位置诊断 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用一轮固定配方 F 和只读诊断，检验保护评分是否提供可利用的位置信息；不把 Fisher 或单状态干预当成精准位置上界。

**Architecture:** 任务起点从当前已合并模型计算历史净残差，并在三个预定任务估计模型分布 Fisher 对角。后续仅临时替换 S 的保护位置、固定 P 和分类头，在独立训练来源样本上比较相同状态、可比更新范数的干预。正式训练始终使用原谱位置，诊断后恢复全部状态。

**Tech Stack:** PyTorch、现有 DualMask trainer、JSON/CSV 实验台账、3090 原项目、Fish lwait。

---

## 状态与授权边界

- 日期：2026-10-12，状态：formal-running；默认关闭的钩子和独立入口已实现并推送。17项专项测试、本地905项轻量测试（15跳过）、服务器896项及GPU短测通过，独立审查与启动门禁复核通过。20:49核验唯一正式F已启动，完整T10尚未结束；提交、PID、日志和启动证据见[运行记录](../experiments/2026-10-12-protection-position-diagnostic-run.md)。
- 先等待固定 P 支持 A/B/C/D 四组完整结束；不根据中间成绩取消组别，不改运行中的服务器代码。
- 本阶段仅规划 3090 一组正式 F，以及一组 GPU 短测。5090、完整残差消融、特权 Fisher 训练及多 seed 均不自动启动。
- 不新建服务器 worktree、不覆盖本机数据 JSON、不保存 checkpoint 或逐任务 QKV/分类头快照。
- 后续真正实施前需再次核验源码、运行身份、待执行队列和用户授权。

## 科学问题与限制

1. W_pre 谱位置、历史残差位置是否给出了不同的信息？
2. 在同一模型状态、相同 S 更新范数下，位置是否改变 Old–New 取舍？
3. 若真实残差区别于打乱残差，这种线索是否值得完整训练验证？

只改变 S 避免直接开放 P 从未受到分类梯度约束的坐标，但 S 也在原谱权限下训练过，仍存在适应偏向。任何负的换位置结果都不能证明坐标保护无用。Fisher 对角忽略跨坐标作用，采样估计存在方差，风险分数不是逐坐标真实损伤贡献。

## 固定训练配方 F

与固定 P 支持队列的 A 配方一致；仅记录开关、prefix、wandb_group 等诊断身份字段允许不同，实施时列出有效配置 diff。

| 项目 | 固定设置 |
|---|---|
| 机器／数据／种子 | 3090，ImageNet-R T10，seed1993 |
| Task0 | 原 unmasked 路径，anchor2.5，20轮 |
| Task1–9 | 20轮；原优化器、学习率、随机冻结 A、B 零初始化 |
| 保护区 | 固定 W_pre 谱评分的 0.90 重要性质量覆盖 |
| S/P rank、γ | 64/64，0.5/0.75 |
| 权限 | S 保护区更新乘0.5；P硬补集 |
| 冲突门 | 完整 packed QKV、S/P 分别精确 Top10% 幅度位置 |
| 冲突强度 | β=min[0.5(1+R_old),1]，R_old=max(0,C_new−C_all) |
| 其他 | 掩码正则0；step；CA5；math-SDPA；不保存权重 |
| 关闭 | 预算α、固定P支持、样本分配、原型位置、教师、梯度初始化、正式训练范数控制 |

这里 α 固定0.5，不采用 energy_equal 的动态预算；β仍保留 NCM 驱动，不固定成0.5。

## 诊断范围与样本

- 只在 Task1/5/9 的任务起点和第2/20轮结束、合并之前诊断。Task0不执行。其他任务照常训练，不插入 Fisher 求导。
- 数据均来自 `source='train', mode='test'` 的确定性视图，不使用旧测试集选评分、位置、归一化或参数。
- 每个旧类按固定样本索引排序：前2张为 Fisher 评分集合，接下来2张为独立旧类干预评估集合。若不足，使用实际可用且互不重叠的样本，记录缺失；不得跨集合复用。
- 当前任务每个新类固定2张作为新类干预评估集合，不参加 Fisher 评分。它们仍可能参与普通当前任务训练，不称为泛化验证集。
- 评分集合与干预评估集合互不重叠；旧训练图片以前可能被训练过，因此也不称为从未见过的旧类数据。
- 记录类别覆盖、样本数、索引哈希、类顺序和预处理指纹。固定独立 RNG，不消耗训练/DataLoader随机序列。
- 正式训练和 CA 仍不使用旧图片。旧图片只用于诊断；结果应同时报告该额外访问与计算成本。

## 任务起点：Fisher 与五套位置

评分时模型处于 eval，当前任务的 B 为零，以前任务已合并；只对12层 QKV临时开启求导，不改变参数值。

诊断概率模型明确为：`p=softmax(scale * cosine_logits_old)`，scale取现有ImageNet-R配置20；仅使用此前已见旧类分类头，不加入当前新建随机头，不使用 CosFace 真标签 margin。

每张评分图片从 `p.detach()` 独立采样2个标签，分别求 `-log p(label)` 对 QKV 的梯度。逐样本、逐采样平方后平均。两次采样分别累积以记录评分稳定性，最终取两者平均；采样标签不反传，真实标签只用于分层取样和后续报告。

这是指定概率模型、有限旧输入集合上的 Monte Carlo Fisher 对角估计，不是精确 Fisher，也不是准确率上界。采用采样标签改变了参考量，但不消除输入覆盖、有限采样、温度及对角近似问题。

历史项冻结为 `R_start = qkv.weight - pretrained_weight`，只指当前任务开始前累计净变化。

五种掩码均按每层 Q/K/V 的原谱保护数量 K 精确选点，评分并列按固定索引处理：

| 名称 | 评分／位置 |
|---|---|
| W | 原谱保护掩码 |
| WR | normalize(I_pre)+normalize(abs(R_start))，λ=1 |
| WRperm | normalize(I_pre)+固定行列置换(normalize(abs(R_start))) |
| MC | 采样 Fisher 对角最高位置；仅诊断使用 |
| Random | 原谱掩码的固定行列置换 |

`normalize`固定复用当前 `_normalize_score` 的 min–max 规则；不用 rank 归一化，也不扫描λ。归一化按完整 packed QKV 计算，随后分别按 Q/K/V 选 K。零残差返回零历史项，WR应退化为原谱排名。记录各项总质量、分位数及混合比例，不把 λ=1描述为等贡献。

WRperm 在 Q/K/V 内置换残差评分，保存分布，不置换谱评分；Random 保留原掩码数量与行列占用的排列结构。置换由独立 seed/task-independent layer 映射生成，跨任务固定，零残差和并列测试采用统一精确选点规则。

各套掩码只用于统计／诊断，不安装成正式训练权限。

## 第2/20轮：风险评分与 S-only 干预

### 风险评分

任务起点 B=0，因此当时的 `F * delta**2` 全为零，不据此排名。

到诊断轮末，用当前 S 的同一 BA、同一原幅度冲突门形成未施加 S 保护的公共更新：

```python
v = gamma_s * raw_s * (1 - beta * conflict_s)
risk = fisher_start * v.square()
```

不使用已经乘原 H 的 S 更新定义公共风险，否则风险排序本身带原位置偏向。不把累计残差 R_start 与当前 S 更新 v 混用。

风险仅解释为任务起点对角 Fisher 对当前更新的近似暴露量；θ和分类头已随训练变化，且忽略一阶项与跨坐标项，不称为实际损伤。输出各掩码覆盖的风险质量、相对随机富集和评分相关性，不按是否像 Fisher 自动筛除候选。

### 实际干预

1. 固定当前骨干、A/B、P有效更新、分类头、β和S幅度门。
2. 每层得到五种 S 有效更新 `u_k=v*(1-0.5*H_k)`。
3. 按每层完整 packed QKV，以五者范数最小值为公共目标，只缩小、不放大；零更新系数取1。
4. 仅临时覆盖 S 有效更新；不得改写 `general_mask`、P补集、P支持或两支共享的诊断位置开关。
5. 在独立旧／新训练来源样本上，以完整已见类别分类头计算 scale20 CE、原始余弦 margin、准确率、预测翻转及对真实标签的改善／退化数量。
6. 另外记录实际未匹配 F 前向，标明与五种范数匹配干预不是严格同强度比较。
7. 恢复全部状态再继续原训练；诊断前后参数、梯度、优化器、调度器、RNG和下一训练步保持一致。

匹配 S 更新范数不保证 S+P 合成范数相同，更不保证函数扰动相同。分别记录 S范数、保持不变的P范数、S/P内积、合成范数和匹配残差。三任务×两时点是六个相关模型状态，不当作六个独立seed。

## 实施任务与验证

### Task 1：配置与样本边界

- Create: `utils/protection_position_diagnostic.py`、`test/test_protection_position_diagnostic.py`。
- 先补失败测试：分层样本不重合、全旧类覆盖、独立生成器、固定预处理、五套掩码精确计数、零残差退化、固定置换与并列规则。
- 最小实现后运行 `python -m unittest test.test_protection_position_diagnostic test.test_protection_position_diagnostic_queue -q`；必须明确它证明样本和评分语义，不证明性能。

### Task 2：Fisher 求导与恢复

- 在同一工具文件实现求导上下文；原 `diagnostic_state` 包含 no_grad，不直接拿来计算 Fisher。
- 用可穷举类别的小线性 softmax 测试 MC 估计的定义及独立采样；精确参考仅用于测试，不在真实模型枚举所有类别。
- 覆盖逐样本平方而非平均梯度平方、采样标签固定、参数/梯度/flags/modes/RNG恢复、旧头范围与scale、失败退出恢复。
- 分别验证两份 MC 图的统计输出，不因稳定性不佳换随机种子或采样到得到理想结果。

### Task 3：只改变 S 的干预

- Modify: `models/attention.py`，增加只在诊断上下文存在的 S 有效更新临时覆盖；默认路径直接保持原行为。
- Test: 前向覆盖仅改变 S；P 的有效更新、掩码、缓存逐值一致；每层缩放不放大；合成范数与内积如实记录；上下文退出原前向恢复。
- Modify: `methods/dlora.py`，默认关闭的 `dual_mask_position_signal_audit` 钩子接入任务初始化后和指定轮末；不得改变优化器参数分组或合并规则。
- Test: 默认与Task0回归；开关开/关同初态下前向、损失、梯度、优化步、合并一致；诊断异常恢复且不继续污染训练。

### Task 4：独立入口与汇总

- Create: `scripts/run_protection_position_diagnostic.py`、`scripts/analyze_protection_position_diagnostic.py`、`scripts/sweeps/protection_position_diagnostic_3090.json`、`scripts/10_12_imgr10_protection_position_diagnostic_3090.sh`、对应队列测试。
- 复用现有启动/指纹/退出码/分析异常机制，不改变固定 P 支持的四组入口。
- 单正式组F；短测Task0一轮、Task1三轮，诊断轮2/3，覆盖诊断之后还有下一训练步及最终合并。短测仅验证运行，正式恢复2/20轮诊断。
- 汇总支持单seedSD不适用、部分结果、旧样本诊断与正式测试区分、三任务六状态完整性检查；分析失败单独记录，训练失败暂停，不自动重试。
- 专项通过后跑适用全量轻量测试、独立代码审查及GPU短测。全部通过才正式训练；不按短测准确率筛选。

## 接续安排

实施完成后提交、推送完整 SHA。3090只使用 `shengqin3090ti` 和原项目 `/home/shengqin/lys/baseline/LoDA_ICML2026`，保留本地JSON。

当前四组仍运行时，仅在项目外写接续脚本，重新核实真实外层队列PID后使用 Fish lwait；不使用本计划中的历史PID，不修改正在运行的代码。退出后先检查四组完整T10、退出码和指纹，再更新、核验、短测和启动新目录中的一次F。若旧队列已结束，直接检查完成记录，不对不存在PID调用lwait。不开定时任务。

普通F最近约65分钟只是训练参考；MC反向传播、额外前向和临时掩码计算另计，不能称为免费离线分析或保证只多几分钟。短测记录成本，启动前给出修正ETA。

## 交付与后续决策

交付一组正式T10的Average/Last/Old/New/StageOld/StageNew/Forgetting和逐任务曲线，以及五种位置的数量/重合/风险覆盖、MC稳定性、六个S-only干预状态的Old–New图、S/P合成范数、成本和配置/源码指纹。只保存统计、临时评分在当前任务结束释放，不保存图片、逐样本梯度、历史大矩阵或模型快照。归档现有实验台账和启动台；不宣称服务正在监听。

- WR相对W及WRperm若在多个任务呈一致Old–New正向线索，可提出三组完整训练验证；新训练需单独确认，不按测试成绩自动开跑。
- 三组预留为W／WR／WRperm，保护数量匹配，α=.5及其他配方不变，S/P一起采用相应新权限。三组同机同提交重新运行，均记录实际更新量；不能把本次S-only诊断当作已验证P权限。
- MC位置若有优势，只说明旧类信息提供了可利用的局部位置线索；如安排MC掩码训练，应单独标为旧数据辅助特权参考，不能并入rehearsal-free方法结果。
- WR与打乱残差接近，不支持历史残差坐标对应关系有独立价值。负诊断只能收缩当前候选，不裁定所有坐标策略无用。
- 正向完整训练的核心配对再规划seed1996/1997，使用三seed配对差；不把六个模型状态或多层记录当seed，不给全部失败候选补多seed。
- 新正式基线不自动替换，不增加λ、归一化、Fisher温度、覆盖或rank扫描；不新增损失、教师、原型分配或P无掩码预热。

## 文献与证据边界

- 当前代码依据：`models/attention.py::_safe_delta/rebuild_dual_masks`、`models/network.py::interface`、`utils/dualmask_core_audit.py::position_diagnostic`、`utils/plora_gradient_init.py::initialize_gradient_a`。
- 现有F依据：`scripts/sweeps/p_fixed_support_3090.json`及`docs/plans/2026-10-12-p-fixed-support.md`，不是ρ=1任务起点位置队列。
- Fisher定义与经验近似局限：Kunstner, Balles, Hennig, *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent*, NeurIPS 2019，https://arxiv.org/abs/1905.12558。
- 研究技能要求分开实现健康、机制生效、完整性能和贡献主张；上述计划不等于实现或实验已经完成。
