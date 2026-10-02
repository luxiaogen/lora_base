# P 冲突选点的方向敏感性：3090 三组完整训练

本轮不是9/25的P/B预条件，不解冻A，不启用合成共享门。
只替换Task1–9的P冲突选点；S保护/冲突、P可塑区、损失系数、合并、CA结构不变。
默认 `p_direction_score=off` 保持原算法。Task0沿用原unmasked路径。

## 预先固定的三组

| 模式 | P候选坐标排名依据 |
| --- | --- |
| coordinate | P可塑区更新的绝对幅度 |
| spectral | W_pre参考基下的加权完整谱能量 |
| signed | 同样谱能量，但去掉对角线正向增强部分，保留削弱和非对角混合 |

完整ImageNet-R T10、seed1993、anchor2.5、20轮、CA5、math-SDPA、原layer/adaptive设置。
不重跑原基线，不保存checkpoint/逐样本特征，不增加教师、回放、类别路由或推理次数。
数据/预训练文件路径仍由3090本机JSON给出。

## 精确评分规则

分别对W_pre的Q/K/V矩阵做完整SVD；基只计算一次并缓存在GPU，不用累积后的权重。
在P可塑区限制之后，令可塑更新为D，W_pre=U diag(s) V^T，C=U^T D V。

```text
e_i = s_i² / sum(s²)
R_ij = C_ij² × (e_i + e_j)/2
spectral: 使用整个R
signed: 将R_ii替换为e_i × min(C_ii, 0)²
坐标分数 = U² × R × (V²)^T       # U²/V²均为逐元素平方
```

此映射是非负谱能量的坐标归因，不是带符号矩阵重建，不声称逐坐标因果责任。
对角增强、削弱、非对角混合均相对于固定W_pre参考基；不是当前权重的奇异值变化，
更不等于增强有益、削弱有害或保护旧类的证明。近简并奇异方向有基选择局限。
若对角增强能量本来很少，signed和spectral可能几乎相同，必须报告而非包装创新。

## 抑制量匹配，不用坐标数冒充强度

在每个候选自己的同一权重处，先按原P冲突规则取参考mask，与P可塑区相交。
每个Q/K/V单独定义目标 `target = beta × ||D × reference_mask||`。
候选在可塑区按新分数取参考的有效K，再令 `beta_candidate=target/||D×candidate_mask||`。
若该强度会超过1，则按排名扩展选点直到能达到目标，再计算强度。
这允许坐标数/强度不同，但移除更新的Frobenius范数在同权重处匹配。
不同组训练轨迹仍不同，不声称每步的绝对target跨运行完全一致。
固定预算也可能迫使signed组选到零风险坐标，因此不声称完全放行所有增强方向。

P前向先乘gamma再计算实际gate，和merge同一计算顺序；正则复用该前向gate，原损失公式不变。
协议/算法断言仅在test，不向训练方法加入防御性验证。

## 启动和输出

```bash
bash scripts/10_02_imgr10_p_direction_score_3090.sh
# 仅短测： --mode smoke；仅正式训练： --mode t10
```

默认先三组各Task0–1×1轮GPU短测，退出0后跑三组完整T10；短测成绩不能用于判断提点。
原3090每个T10约62分钟；谱矩阵变换新增成本，正式时长尚待实测，先预留约4–6小时。
没有自动按测试分数筛选、截断或扩展参数网格。

`logs/shell_logs/imgr10_p_direction_score_3090/<timestamp>/`：
每组run.json记录提交/有效配置/本机路径/源文件SHA/命令；training.log记录训练；
主目录queue.json记录退出码与时长，results.csv/json记录正式成绩，direction_diagnostics.csv按任务/层/QKV记录实际选点、强度、参考与实际移除范数、三种谱能量占比及同权重处signed/spectral实际掩码Jaccard。
不使用会重新选择旧P掩码的日志冒充实际门控。诊断不额外保留稠密矩阵文件。

## 判断边界

先检查完整10任务、相同Task0、实际移除范数误差和前向/merge一致性。
signed优于coordinate但不优于spectral，只支持换表示空间，不支持方向符号有额外价值。
只有signed整体优于两种幅度控制，且Old/遗忘不恶化、New改善，才值得补进一步验证。
单seed筛选不是统计显著；旧基线仅作参考，不能把不同提交/诊断开关当严格配对。
本轮是尚未证明有效/新颖性的候选，不是现成安全更新方法。
