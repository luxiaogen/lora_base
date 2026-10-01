# 两个固定输出的互补上界：3090

只跑 ImageNet-R T10、seed1993 一轮，不改训练方法。anchor2.5、20轮、CA5、math-SDPA、原layer双门。
本机 JSON 提供 data_path、device 和预训练文件；不保存权重。没有访问或启动服务器。

## 问题与上界

每个任务CA完成后，对同一批图片、全部已见类别，比较：

- 原模型：当前合并ViT + 当前全部已见分类头。
- 固定参照：不可变的AugReg预训练ViT W_pre + 已有的W_pre-NCM原型。
- 互补Oracle：两个输出至少一个预测正确，就计正确。
- 真任务ID范围参照：原模型只在样本所属任务的类别中选择，单独报告。

互补Oracle是**这两个固定输出之间逐样本选择的准确率上界**，包含原输出，准确率不会降低。
它用真实类别标签判断哪个正确，权限强于只知道任务ID；不是正式任务无关推理成绩，不是整个方法的理论上界。
不保证高于历史P-control Oracle。历史P候选并集、W0原型路由做过，不能把“候选并集”概念当新贡献。
这次测量不同的两个专家，并将纠正、破坏的样本逐个保留下来。

## 合法信号及失败条件

两种专家的输出均为全已见类余弦分数，类别顺序一致。仅在预测不同的样本上，预先固定：

1. margin_advantage：NCM最高与次高分差 − 原模型最高与次高分差；大于0选NCM。
2. proposal_advantage：NCM对其候选相对原候选的支持 − 原模型对其候选相对NCM候选的支持；大于0选NCM。
3. base_uncertainty：原模型前两名分差的负值，只作为风险排序诊断，不部署阈值。

信号不读取真实标签或任务ID；阈值不按测试分数拟合。**这些分数未经过跨专家置信度校准**，只是待检验代理。
标签只在报告时统计救回、破坏与AUC。AUC仅比较“恰有一个专家正确”的分歧样本，排除两者都错的样本；
同时报告两者都错的分歧数和被规则选中的数量，避免选择偏差。关键是净收益 `(救回−破坏)/全部样本数`，不是只看救回数。

每任务额外报告当前训练集的相同指标；原型与专家见过这些图片，**不是expert-unseen holdout**。
不读取旧训练图片，不从test选择规则，也不保存旧图。旧原型是各类到来时由其训练数据计算的既有768维向量。

- 若互补空间很小，停止用这两个输出逼近上限。
- 有空间但信号不能区分救回/破坏，不能声称合法替代已找到。
- AUC大于0.5也不足以证明有效：固定规则必须在Old/New净取舍上有实际收益。
- 即使规则有收益，正式使用两个骨干输出也有额外推理成本；本轮没有改变正式推理，更没有证明保持一次前向的方法已实现。

## 命令（Fish交互；脚本为Bash）

默认先2任务、各1轮短测，再一个完整T10；短测失败或缺诊断文件即停止，**不设准确率一致性断言**。

```fish
lrun scripts/10_01_imgr10_two_expert_oracle_3090.sh ./logs/10_01_imgr10_two_expert_oracle_3090.log
```

仅查看展开命令：

```fish
bash scripts/10_01_imgr10_two_expert_oracle_3090.sh --dry-run
```

已独立完成短测后，可用 `--t10` 仅跑完整组。每次启动新时间戳，不覆盖旧结果。
基于最近3090原配方约78–82分钟，含新增离线前向暂估1.5–2小时；新增开销尚未实测。

## 输出及复现边界

外层日志打印完整代码提交、展开命令与诊断目录。目录形如：

```text
logs/shell_logs/imgr10_two_expert_oracle_3090/<timestamp>/
  smoke|t10/run.json             # 提交、源文件哈希、实际配置（含本机路径）
  smoke|t10/training.log
  smoke|t10/diagnostics/
    task_00_test_report_only.json/csv
    task_00_current_train_seen_probe.json/csv
    ...
    task_metrics.csv              # 每任务Total/Old/New、两种规则、纠错/破坏、AUC
    summary.json                  # Average/Last和实际完成任务数
```

CSV只有样本索引、标签、预测与几个分数，不存logits矩阵、特征、历史P分量或checkpoint。
原W_pre评估路径保留；新增前向使用no_grad/eval并恢复Python/NumPy/PyTorch随机状态、模型模式和anchor权重。
测试覆盖真实Attention的开关前后权重、logits、随机轨迹一致；本地测试不能代替CUDA短测或完整性能结果。
