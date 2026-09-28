# anchor2.5 两机确认队列：待服务器运行

## 两次运行，不重跑基线

| 机器 | 候选 | 范围 | 本机历史参考 | 预计 |
|---|---|---|---|---|
| 3090 | 独立Task0 anchor2.5 | 完整T10，seed1993 | 9_27_imgr10_head_balance_3090.log的head_balance_off、anchor10，Average87.000/Last82.48 | 80–90分钟 |
| 5090 | 独立Task0 anchor2.5 | Task0–2，seed1993 | 9_27_imgr10_head_balance_5090.log的head_balance_off、anchor5，前三任务[96.81,92.28,89.93] | 约50–60分钟，共享GPU可能更久 |

这不是两seed验证；两台各与本机历史参考比较，不跨机相减。
历史参考与当前代码提交不同，运行后仍须核对有效配置/环境，不能声称严格同提交配对。
3090检验T3收益是否延续到后续九任务；5090检验相同anchor改动在另一环境是否也有效。
5090仍是T10类别顺序，只提前止于Task2，不把total_sessions改成3。

两机均从Task0正常训练，不隐含加载checkpoint。相对各自参考只改anchor到2.5：
20epochs、CA5、math-SDPA、S/P学习率0.02、双门、掩码正则0.01、冻结增量A均保持。
全量原训练数据/官方测试，关闭holdout、新分类头准备、蒸馏等候选；
不自动按测试成绩筛选/续跑，不扩扫参数，不加入训练中的assert或配置验证。
读取本机exps/dlora/imgr10.json的数据路径和设备，所有实验覆盖项显式记录。

## 命令

两台先在原项目拉取本分支：

```fish
git switch codex/mask-budget-comparison-20260924
git pull --ff-only origin codex/mask-budget-comparison-20260924
git rev-parse --short HEAD
```

若Git提示本地改动冲突，停止更新，不覆盖数据路径或本地修改。

可选通路短测（不是正式结果，也不会自动进入正式训练）：

```bash
bash scripts/9_28_imgr10_anchor2p5_t10_3090.sh --smoke
# 5090改用对应的t3脚本，同样支持--smoke
```

3090正式一次：

```fish
lrun scripts/9_28_imgr10_anchor2p5_t10_3090.sh ./logs/9_28_imgr10_anchor2p5_t10_3090.log
```

5090正式一次：

```fish
lrun scripts/9_28_imgr10_anchor2p5_t3_5090.sh ./logs/9_28_imgr10_anchor2p5_t3_5090.log
```

脚本支持--dry-run；记录commit、工作区状态、完整运行命令及退出码。
每次内层日志位于logs/shell_logs/<同名实验>/<timestamp>/anchor2p5.log。
没有自动GPU短测/单元测试门禁，不动已经发布的旧队列或共享默认JSON。

## 结果判定

- 3090：完整Average/Last、Task1–9平均Total/Old/New、最终遗忘；不能只因Task0提高就说增量性能改善。
- 5090：Task0、T1/T2 Total和Old/New、三阶段Average、Task2遗忘；不能与3090绝对分数比较。
- 如果后续收益消失或Last下降，不继续密集扫描anchor；若后续净收益保留，再确认稳健性。
- anchor2.5已按探索性官方测试选出，须披露选择过程，不能把这些同seed结果包装成独立泛化验证。

## 本地验证状态

18项新队列/相邻回归通过；两JSON由研究技能生成器分别展开1次，Bash语法和两脚本dry-run通过。
验证只证明队列配置与命令正确，不证明CUDA通路或提点。没有连接、启动任何服务器。
