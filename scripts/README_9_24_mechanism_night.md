# ImageNet-R seed1993：冲突机制与固定混合粒度

只在当前项目执行，不创建 worktree/新项目。分支 `codex/mask-budget-comparison-20260924`。
脚本读取当前机器 `exps/dlora/imgr10.json` 的 data_path；不修改任何数据集 JSON。
共同使用 AugReg ViT-B/16、QKV、fresh Kaiming、Task0/后续任务20轮、CA5、math-SDPA。
所有比较均为完整 T10、seed1993，结果只支持单 seed 初筛。

## 运行顺序

3090（新增6次，预算约8–9小时；诊断开销以实测为准）：

| 组 | 冲突正则 | S冲突门 | P冲突门 |
| --- | --- | --- | --- |
| A | 开 | 开 | 开 |
| B | 关 | 开 | 开 |
| C | 开 | 关 | 关 |
| D | 关 | 关 | 关 |
| E | 关 | 开 | 关 |
| F | 关 | 关 | 开 |

保持 S protection、P plastic mask、保护正则0.01、Energy50/floor10和旧重叠自适应强度不变。
开关同时控制训练前向和最终合并，不关闭冲突损失本身。C组仍计算冲突惩罚。
A–D为2×2机制对照；B/D/E/F拆分S/P门贡献。并非新方法声明。

```bash
lrun scripts/9_24_imgr10_gate_reg_overlap_3090.sh ./logs/9_24_imgr10_gate_reg_overlap_3090.log
```

5090D：先等当前三分数脚本结束，再启动新增4次，**不要同时抢同一张GPU**：

```bash
lrun scripts/9_24_imgr10_fixed_rho_5090.sh ./logs/9_24_imgr10_fixed_rho_5090.log
```

新增顺序：model/ρ=0 → mixed/ρ=0.5 → mixed/ρ=0.25 → mixed/ρ=0.75。
固定冲突分数、关闭冲突正则、关闭能量/旧重叠自适应，与三分数实验的共同设置一致。
预计新增约3.3–4小时；加已有三分数约2.5–3小时，总预算约6–7小时。
全局Top-K额外耗时尚无当前GPU实测，不能保证硬性7小时截止。

仅当三分数实验尚未启动，才运行全部7次的入口：

```bash
bash scripts/9_24_imgr10_mechanism_full_5090.sh --from-scratch
```

## 预算与旧layer基线复用

ρ表示局部分配比例。每层从局部参考mask中保留floor(ρ K_l)个最高分坐标；
余下K−Σfloor(ρ K_l)从未选坐标跨层竞争；绝不重复选择，不产生软mask。
ρ=0严格调用原model选择器，ρ=1原样返回局部参考mask。

“同预算”指同一组分数输入下匹配原局部参考mask实际K。原实现用阈值处理相同分数，
可能超过名义10%，全零更新也可能选0；本轮不静默修改这个历史行为。
不同训练轨迹的实际K仍须核对，不能只凭conflict_ratio=0.1声称跨run严格等预算。
新增 AppliedConflictBudget 日志输出每任务/层/分支的reference_k、applied_k、QKV密度和merge_error。
混合/全局应按任务与分支跨层求和比较预算，不要求每层保持同样数量。
旧layer日志只有在有效配置一致时才能复用。若旧日志缺乏实际K证据，论文级严格预算声明需另补有统计的layer参照，不能臆造。

## ΔW统计：不等于JANUS逐步梯度诊断

3090在每次真正合并前读取 `gamma * B @ A` 和实际safe_delta。
逐S/P、层、Q/K/V比较同一分支历史任务，Task0没有P不补假数据。
记录Frobenius方向余弦（带符号）、输入侧Gram相似度、有效秩、raw/safe/removed范数。
Gram相似度是 `cosine(D_i.T@D_i, D_j.T@D_j)`，是能量加权谱相似度，
**不是严格的主角度/无权子空间重叠率，也不能和随机A的8.33%直接横比**。
有效秩为trace(Gram)^2 / ||Gram||_F^2；零更新返回null，不写成正交或0重叠。

缓存CPU FP32更新及Gram到每run独立临时目录，无新增GPU历史或state_dict参数。
T10/12层/d768峰值临时磁盘约6GB，建议预留8GB以上。完成最后任务或正常对象释放时清理；
强制kill可能遗留临时目录。JSONL标量保留，清理缓存不影响训练checkpoint。
日志中diagnostic_seconds计入运行总时间；不应把带诊断的总耗时直接作为部署训练效率。

## 检查与短测

```bash
bash scripts/9_24_imgr10_gate_reg_overlap_3090.sh --dry-run
bash scripts/9_24_imgr10_fixed_rho_5090.sh --dry-run
python -m unittest test.test_night_mechanism test.test_night_mechanism_scripts test.test_global_conflict_budget
```

服务器短测（所有组各跑Task0–1、每任务1轮，仍会读真实数据；不是正式结果）：

```bash
bash scripts/9_24_imgr10_fixed_rho_5090.sh --set max_tasks=2 --set init_epoch=1 --set epochs=1 --set ca_epochs=1 --set wandb_mode=disabled
```

不要把短测命令附加到正式夜间命令。每run前缀带时间戳，短测不覆盖正式日志。
launcher只检查当前机器JSON的路径，不在训练方法中添加系统式参数防御。

## 汇总

只给汇总器传每次训练独立日志，不传含多次训练的外层日志：

```bash
python scripts/summarize_night_mechanism.py logs/shell_logs/imgr10_gate_reg_overlap_3090/*.log --output analysis/night_3090
python scripts/summarize_night_mechanism.py logs/shell_logs/imgr10_fixed_rho_5090/*.log --output analysis/night_5090
```

生成summary.csv（完成状态/Avg/Last/Old/New/遗忘/时间/训练参数区间）、
overlap.jsonl（层×任务对证据）、applied_budgets.jsonl（层×QKV密度与真实预算）。
LoRA训练和CA优化参数分别报告，不相加成同一阶段参数量。
曲线/热图和性能结论待真实日志完成后生成，不提前声称优化有效。
此次不实现可学习ρ、梯度修正或新Task0表征损失；这些须根据本轮证据单独设计。

## 后续论文启发的适配边界

- [JANUS-LoRA](https://arxiv.org/pdf/2605.28495)：论文讨论A/B共同优化。当前代码Task0训练S的A/B，Task1起S/P冻结A只训练B。
  固定A时原始单步更新准确地是gamma×δB×A，没有B×δA项；门控变化另算。
  本轮任务末累计ΔW相似度不能证明逐优化步的目标方向偏离。
  下一步若研究修正，应先定义全矩阵目标、测量冻结A可实现性及实际一步更新误差，包含SGD momentum影响。
- [DGS](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DGS_Dual_Gradient_and_Semantic-Shift_Guided_Low-Rank_Adaptation_for_Class_CVPR_2026_paper.html)：梯度融合值得参考，但我们已在前向和merge使用门控，不能声称从“仅merge处理”首次转到训练。
  QKV坐标mask为[3d,d]，B梯度为[3d,r]，不能直接逐元素相乘。
- [RSIAT](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Representation-Steered_Incremental_Adapter-Tuning_for_Class-Incremental_Learning_with_Pre-Trained_Models_CVPR_2026_paper.html)：Task0表征引导可作为独立候选；当前分类损失是CosFace而非普通CE。
  不以测试Task0最高值挑选权重；必须验证完整T10的旧类保持与新类学习。

先看本轮同机配对结果，决定是否保留冲突正则和哪一分支门；不能由“更新重叠高”直接推出需要正交约束或某篇论文一定会提升。
