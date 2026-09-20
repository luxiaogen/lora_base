# P-conflict Q/K/V 分组诊断（不选择 gate）

分支从 87f7286 建立；main 不变。只跑 ImageNet-R T10 seed1993，
QKV、rank64、20 epochs、CA5、Task0 unmasked+anchor10，关闭额外区域减量。
从头训练，不加载历史 Task0 checkpoint。JSON 的 data_path 不覆盖。

## 测量内容

- 正常合并时，额外保存当前任务 P 的 `safe_delta * conflict_mask`（已经过原保护/抑制及 gamma 缩放），CPU 暂存。
- 每层 Q、K、V 各一组；12层共36组。每次完全移除一组当前任务贡献，然后精确恢复。
- 不移除整个 Q/K/V 更新，不更改 S，不恢复原 suppress 丢弃的更新，不干预历史任务分量。
- Task1、Task2、Task9：合并后 CA 前测当前训练样本；CA 后测已见类别测试样本。
- 每类别按固定位置取最多4张，测试变换、顺序固定、不消耗持久 RNG；最终正式评估仍是完整测试集。
- 输出 old/new accuracy、task accuracy、corrected/broken、local/cross/global margin change 的均值/中位数/Q25/Q75/正值比例。
- `margin_change = 移除后 - 原基线`，正数表示此次移除对测量样本有帮助。
- 输出每组非零坐标数、移除范数。各组范数不同，不能把原始效果大小当作等预算因果比较。
- 每次诊断结束重新计算基线 logits，要求误差不超过1e-6；状态/RNG/异常恢复另有单元测试。
- 每任务输出 acquisition/current/peak 和 drop；区分学得少与保留得好。

训练子集不是独立 holdout。测试标签仅用于诊断，不更新任何权重、选择 gate、设置最终预测。
CA前后两种数据源不同，不能用两组差异推断 CA 的因果作用。
这是小样本探索，既非正式改进成绩，也不能证明删除该组后重训/重做CA仍会改善。
本轮不生成“有害组清单”供训练自动使用；若观察到稳定结构，另定训练验证协议。

## 5090 执行

在仓库根目录、已激活训练环境时：

```bash
env CHECK_ONLY=1 bash scripts/9_20_p_conflict_groups_5090.sh
lrun scripts/9_20_p_conflict_groups_5090.sh ./logs/9_20_p_conflict_groups_5090.log
```

CHECK_ONLY 仅为轻量正确性检查，不是完整GPU训练 smoke。
训练基准约47分钟；36组在三个阶段的重复推理未实测，暂留1.5–2.5小时，不保证1小时结束。
可以搜索 `P-conflict groups` 和 `P-group retention` 定位诊断及学习/保留曲线。
不需要安装其他第三方仓库或切换到 CL/SD 迁移环境。
