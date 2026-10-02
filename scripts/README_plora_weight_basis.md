# 当前分类梯度＋现有权重先验：冻结 P-A 三任务筛选

这是训练方法候选，不是已验证的提点，也不是恢复历史特征二阶矩。
任务开始时用当前任务训练图的确定性视图取 4 个批次（最多 192 张），只计算分类损失梯度。
不执行优化器、不预训练分类头、不读取旧图片或测试图片。
临时允许 QKV 权重求梯度，但不修改它；随后恢复 requires_grad、已有梯度、模型模式和随机状态。
当前分类头尚未训练，因此这组梯度只是初始化时的需求线索，是否足够可靠由实验验证。

## 三组只改变 Task1+ P-A 的输入方向

| 顺序 | plora_a_init_mode | 做法 |
| --- | --- | --- |
| 1 | random | 保留原 Kaiming A，执行相同的只读梯度采样作为过程对照 |
| 2 | gradient | 按 P 可塑区内分类梯度的右奇异方向选取 rank 个方向 |
| 3 | weight_prior | 从相同梯度候选方向中，优先保留学习信号强且权重先验能量较低的方向 |

每层 G 为当前分类损失对 QKV 权重的平均梯度，再乘原 P 可塑区掩码。
W_pre 是不可变预训练 QKV，R 是当前任务开始前 qkv.weight − W_pre。
G 的右奇异向量 v 与奇异值 s 给出候选，联合组固定评分为：

```text
e_pre(v) = dim × ||W_pre v||² / ||W_pre||_F²
e_old(v) = dim × ||R v||² / ||R||_F²
score(v) = s² / (1 + e_pre(v) + e_old(v))
```

两个能量先验各自归一化，零能量用数值下界；不扫描系数。
该分数只表示权重空间的候选保护代理，不是旧数据敏感度；R 也可能包含抵消或不重要的变化。
第三组必须优于 random 和 gradient，才有依据说两个权重先验提供了额外价值。

为避免把尺度或条件数变化当作方向优势，候选不仅匹配 A 每行范数，还保留整个 A Aᵀ。
具体为原 A 分解成 U Σ Vᵀ，再令 A_new = U Σ V_selectedᵀ。
A 冻结、B 零初始化、P rank 与原自适应设置保持一致。
Task0、S、双门选点与抑制、正则、CA、分类头训练和正式推理均保持原样。

## 正式比较和诊断边界

ImageNet-R、seed1993、完整训练集、正式测试 Task0–2，anchor2.5、20 epochs、CA5、math-SDPA。
这不是完整 T10 或多 seed 结论。三组均不保存 checkpoint；数据路径和 GPU 读取本机 JSON。
随机组在本次提交下重跑，是必要的同流程对照；不另重跑完整 T10 基线。

每层仅记录小型标量：梯度能量、W_pre/R 能量、A Gram 误差、采样摘要、初始化耗时。
合并前复用真正要写入模型的 P 更新，记录原始/实际更新范数、偏离 A 输入子空间的比例，
以及与 W_pre/R 输入权重能量的重叠。逐坐标门可能使更新离开原 A 子空间，不能凭 A 声称安全。
这些权重代理变小不等于 Old 改善；判定仍以本机三组的 Old、New、Average、Last、Forgetting 为准。
当前 CA 仍保存类别协方差，因此只能说“不用历史激活统计构造 A”，不能说“整个方法无二阶统计”。

只有联合组在 New 提高同时保住 Old、且 Average/Last 有净收益时，才值得另行确认完整 T10。
如果 gradient 已有收益而联合组无收益，则只支持普通梯度初始化，不能宣称保护先验贡献。

## 运行

```bash
bash scripts/10_02_imgr10_plora_weight_basis_t3_3090.sh --mode dry-run
bash scripts/10_02_imgr10_plora_weight_basis_t3_3090.sh
```

默认先顺序执行三组 1-epoch Task0–1 GPU 短测，全部成功后才执行三组正式 Task0–2。
短测成绩不是性能证据。预估三组正式训练约 65–80 分钟，短测与额外初始化约 10 分钟；
依据上一队列完整 T10 每组约 75–77 分钟，实际耗时取决于任务样本数和 GPU 负载。

结果输出到 `logs/shell_logs/imgr10_plora_weight_basis_t3_3090/<timestamp>/`：
每组 training.log、run.json（提交/有效配置/源文件哈希）、queue.json、results.json/CSV、
basis_initializations.csv 和 basis_actual_updates.csv。没有特征/旧图片/历史矩阵缓存。
