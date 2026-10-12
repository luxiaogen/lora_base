# 3090单组保护位置诊断

在原项目、原 Conda 环境中运行：

```bash
bash scripts/10_12_imgr10_protection_position_diagnostic_3090.sh
```

入口只允许 `imgr10_F_seed1993`。先完成Task0一轮、Task1三轮短测，核验两次诊断及诊断后的训练步，再自动启动一次完整T10。短测或训练失败保留记录，不自动重试；分析异常单独记录。

F使用原谱位置、0.90重要性质量覆盖、rank64/64、固定保护强度0.5、精确Top10%幅度门，以及 `min(0.5*(1+R_old),1)` 冲突强度。固定P支持、预算保护和样本分配关闭；20轮、CA5、Task0 anchor2.5、math-SDPA，不保存权重。

正式训练始终使用原谱位置。Task1/5/9起点计算采样Fisher与历史残差掩码；第2/20轮临时比较五套等S范数位置，只改变S，P和分类头不变。旧训练来源图片仅用于诊断，不用于正式训练或CA；不使用测试集构图。Fisher不是准确率上界，换位置仍有训练适应偏向。

新运行目录在 `logs/shell_logs/protection_position_diagnostic_3090/`，保存manifest、queue、active、有效配置、源码与环境指纹、逐任务成绩和诊断记录。只保存统计，不保存图片、逐样本梯度、历史快照。

```bash
python scripts/analyze_protection_position_diagnostic.py RUN_DIRECTORY
```

结果包括 `results.csv/json`、`tasks.csv`、各 `PositionSignal*.csv/json`、`matching_issues.json`、`report.md`及三张PDF图。原始日志不完整或短测/正式指纹不匹配时不标记有效结果。单seed标准差不适用；五套位置不是五组训练。

在前台执行时按 `Ctrl+C` 停止；后台使用既有Fish `lrun`，启动后以manifest、active和`ps`核对实际PID，不复用历史PID。
