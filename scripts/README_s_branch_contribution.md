# S/P 分支贡献诊断

在实验分支 `codex/s-branch-contribution` 的仓库根目录运行：

```bash
lrun scripts/9_12_s_branch_contribution.sh ./logs/9_12_s_branch_contribution.log
```

数据目录只从 `exps/dlora/imgr10.json` 读取。脚本固定 ImageNet-R、seed 1993、Task0–2、rank 64、20 epoch、CA5，以及当前 S+P 基线的 DualMask 设置。`--dry-run` 打印完整命令而不训练；`--check` 只跑相关单测。

Task1 和 Task2 训练完成、merge 和 CA 之前，对同一组已训练参数及同一批已见类测试样本分别评估 `both`、`s_only`、`p_only`。每组记录 Total、Old、New 和 Task Prediction；随后恢复原来的 `both` 前向，继续正常 merge、CA 和最终评估。这个消融只临时关闭当前任务的一个分支；旧任务已经 merge 到主干的权重始终保留。不能把这里的 `p_only` 解释为“只训练 P”的实验，也不能用诊断结果选择 merge 或更新权重。测试集只用于分析，不参与训练。

日志检索：

```bash
rg 'pre-merge branch contribution|CNN top1 curve' logs/9_12_s_branch_contribution.log
```
