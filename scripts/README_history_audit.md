# 固定类别范围与候选扩张分解（3090）

一轮 ImageNet-R T10 seed1993 基线，anchor10、20轮、CA5、math-SDPA。
复用 head_balance_3090 sweep 的 common_overrides，但 head_balance_weight=0。
数据路径仍读机器 JSON。只读诊断，不是优化候选，不使用测试结果调参。

每个任务 post_ca 保存已见测试样本的 logits、标签、样本索引作为参考。
后续任务三个时点复用 StageAudit 的同一次前向，并保存原始输出。
对每个历史参考的同一批样本，比较：

- reference_correct：当时全局正确数量。
- fixed_correct：现在限制在当时已见类别内的正确数量。
- full_correct：现在全部已见类别内的正确数量。
- fixed_broken/corrected：固定类别范围内，原来对现在错／原来错现在对。
- expansion_only_errors：固定范围正确，但扩大候选后错误。

精确分解：reference_correct - full_correct = fixed_broken - fixed_corrected + expansion_only_errors。
固定范围属于诊断 oracle，不替代正式全局预测。不同历史参考包含重叠样本，不相加。
固定范围退化不能单独归因于骨干；分类头也可能变化。该分解是计数分解，不是候选类别增多的独立因果估计。
最新历史参考 post_ca 到下一任务 pre_merge 包括整个训练阶段；pre_merge 到 post_merge 是合并；之后是统计提取与 CA。

现有 StageAudit 只存汇总，无法离线恢复逐样本轨迹；本轮 pt 文件避免后续重复训练。
输出 logs/history_audit/3090_<timestamp>/taskXX_<stage>.pt，CPU float32 logits，无图片。
T10 全量三阶段 logits 约106 MiB，参考缓存约36 MiB；不新增评估前向、不改变优化器。
外层日志包含 HistoryAudit JSON。没有训练过程 assert 或自动测试阻断。

运行：`lrun scripts/9_27_imgr10_history_audit_3090.sh ./logs/9_27_imgr10_history_audit_3090.log`
可选短测：`bash scripts/9_27_imgr10_history_audit_3090.sh --smoke`（不用于性能判断）。
