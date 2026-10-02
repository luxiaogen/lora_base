# W_pre 关系评分与 Q/K 联合干扰，3090 六组

这是评分候选与机制检验，不是已证明有效的新方法。只改 Task1+ 的 P 冲突排名。
S、Task0、CA、分类头、P 可塑区、训练正则权重和最终推理协议保留。

## 六组完整训练

| p_direction_score | 排名依据 | 范围 |
|---|---|---|
| coordinate | abs(P 可塑更新) | Q/K/V，新的同批幅度对照 |
| wpre_product | 原 W_pre 重要性 × abs(P 可塑更新) | Q/K/V |
| wpre_input | 对 W_pre 输入 Gram 扰动的正向局部贡献 | Q/K/V |
| wpre_output | 对 W_pre 输出 Gram 扰动的正向局部贡献 | Q/K/V |
| wpre_qk | 对 W_pre 逐 head Q/K 联合核扰动的正向局部贡献 | Q/K；V 用 coordinate |
| task_qk | 相同 Q/K 规则，但参照任务开始时的累计骨干权重 | Q/K；V 用 coordinate |

固定 ImageNet-R T10、seed1993、anchor2.5、20 epochs、CA5、math-SDPA。
训练集不划分或减少；无旧图回放、教师、任务ID路由、新损失和权重保存。
JSON 不覆盖本机数据/预训练文件路径。wpre_product 复用现有 SVD 重要性，
并不是“预训练权重绝对值”；其重新选点也不等于恢复全层原门。
旧 spectral/signed 负结果不重跑。

## 数学与边界

令 W 为不可变 W_pre（task_qk 为当前任务开始的 qkv.weight），
C 为已合并历史权重相对 W 的偏移，加本步 S 分支实际经过原保护/冲突门的固定更新，
D 为带 gamma 的 P 可塑更新。因此 W+C+D 对应真正的当前层权重，不能漏算历史增量。
task_qk 的历史偏移为零。C 只作为评分上下文，不改变 S 的训练路径。

输入关系使用 R_in = ||(W+C+D)^T(W+C+D) - W^TW||_F² / ||W^TW||_F²。
输出关系使用 R_out = ||(W+C+D)(W+C+D)^T - WW^T||_F² / ||WW^T||_F²。
分别按 Q/K/V 计算。它们是均匀输入几何代理，不是历史输入分布上的损失。

Q/K 逐 head 使用 H = W_Q^T W_K，R_qk 是联合核差平方除以参考核能量的 head 均值。
偏置用输入齐次坐标扩展纳入，P/S 的新增偏置为零。它在固定层输入下描述
attention logits 的双线性/仿射关系，不等于 softmax attention 距离或整网旧类损失。
上游输入、V、后续层变化仍可能导致遗忘。

关系候选取 score_ij = max(0, D_ij * dR/dD_ij)。
这是“缩小该坐标对本地代理的一阶影响”，保留正负及投影交互，不是 abs(gradient)。
梯度用解析式计算并 detach，只参与排名，不对真实分类梯度做二次反传。
有限幅度、同时抑制多坐标可能破坏补偿，R 不保证降低。
零/负贡献也可能因范数目标而被纳入，应结合日志解释。

六组按每层每个 Q/K/V 匹配“同权重处原 P 门 × 可塑区”的移除更新范数。
初始选点数沿用该交集，若不足以供应目标范数便按排名扩展，并调节抑制强度。
这是等移除范数，不是等选点数、等强度或跨训练轨迹等绝对范数。
日志同时记录实际 K、强度、目标/实际移除范数、局部正贡献比例及 merge_error。
前向和合并调用同一实际门；原掩码正则复用该门，原正则评分和归一化不更换。

## 机制证据

每组在 Task1、5、9 的训练结束、合并/CA之前做固定权重的六门反事实评估。
同一训练状态、每组最多64旧/64新测试图片、相同全部已见类别分类头。
样本按固定类别均衡规则选取，只在本次评估内保留输入；不写图片、特征或logits缓存。
临时替换所有层 P 门，S 和全部模型权重不变。
比较全局准确率、旧/新类 margin、相对该模型原实际门的 corrected/broken，
以及各层实际移除范数、输入/输出 Gram 与 Q/K 核代理。
JSON/CSV 的 reference_mode 明确标记比较基准；不输出真值分区 Oracle 指标。
原模式、门、RNG 恢复后再合并。
测试标签只用于样本分组和报告，不能进入选点、优化、超参选择或队列调度。

这能回答“相同模型中换门是否改善旧/新表现”，以及代理与这些改变是否一致。
它不能单独证明逐优化步的因果机制、推断历史所有输入的安全性或给出统计显著性。
最终 T10 结果仍需独立比较，不能把 counterfactual 的最高分当正式成绩。

预先固定的判读原则

- 机制方面，在同一固定权重处，与 coordinate 的 accuracy/margin 直接配对比较，
  旧类改善且不持续伤害新类。corrected/broken 均相对该权重原实际门，不能跨模型直接相减。
- 性能方面，Average/Last有净收益，Old/New不是简单交换，Forgetting不能明显恶化。
- 若代理更小而旧类仍更差，不能将关系保持包装成旧知识保持。
- 若 task_qk 优于 wpre_qk，只支持任务开始参考的价值，不支持不可变 W_pre 更优。
- 若与幅度持平或更差，保留负结果，不自动追加权重/预算扫描。
- 单 seed 是初步证据；旧类测试只作报告，正式选配方须另做训练集验证。

## 运行与输出

```bash
bash scripts/10_02_imgr10_p_functional_score_3090.sh
```

默认先六组 Task0/1各1轮、CA1短测，全部成功才顺序运行六组完整T10。
也支持 `--mode smoke`、`--mode t10`、`--mode dry-run`、`--only wpre_qk task_qk`。
不根据测试性能跳过或挑选候选；失败保留日志并停止，避免错误继续整夜。

`logs/shell_logs/imgr10_p_functional_score_3090/<timestamp>/` 保存 manifest、逐组
run.json（完整有效配置、源文件SHA256）、training.log、queue.json、results.csv/json、
direction_diagnostics.csv、mechanism_interventions.csv。原始日志保留逐层代理。
当前仅生成原始证据/CSV，未声称图表或论文机制已经成立。
刚完成的 coordinate/spectral/signed 每组约73–75分钟，本轮六组预留8–10小时；
实际新增矩阵运算和机制评估开销以本轮GPU日志为准。
