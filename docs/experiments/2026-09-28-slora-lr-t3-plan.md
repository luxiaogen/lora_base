# 5090：仅降低增量 S 学习率（待运行，不是性能结论）

## 问题与边界

已有双门和P学习率实验显示稳定性/可塑性取舍；分类头竞争和蒸馏也未带来稳定净收益。
这些结果不能证明S更新是瓶颈。本次只检验一个未找到已完成记录的简单假设：
保留P学习能力，减小S每一步更新，能否在不伤New的情况下改善Old和Total。
这是分支学习率消融，不是已验证的新方法，也不是梯度方向修正。

## 固定配方与唯一改动

ImageNet-R，seed1993，5090，anchor5，Task0/增量20轮，CA5，math-SDPA；
T10类别顺序（init_cls=increment=20、total_sessions=10），仅完成Task0–2。
使用全部原训练数据，关闭incremental_holdout/task0_validation；官方测试仅报告预先固定的候选。
没有测试分数自动筛选、早停或T10续跑。原W0训练集competence统计保持不变。

| 阶段 | S初始学习率 | P初始学习率 | 当前分类头初始学习率 |
|---|---:|---:|---:|
| Task0 | 0.02（不改） | 原本不启用 | 0.02 |
| Task1–2 基线 | 0.02 | 0.02 | 0.02 |
| Task1–2 候选 | **0.01** | 0.02 | 0.02 |

仅增加 `slora_lr_multiplier=0.5`，余弦调度时保持此比例。
Task1以后原有A冻结设置不变，因此候选作用于当前S的可训练B；不解冻历史LoRA。
S/P门控、gamma、掩码评分/预算、正则0.01、merge、分类头初始化和CA都不改。
默认 multiplier=1 保留原optimizer参数分组；Task0忽略此开关。
没有更改默认JSON，data_path和device仍读取5090本机JSON。

## 历史参考，不重跑基线

原日志：`/Users/luxiaogen/Desktop/loda_logs/9-27/9_27_imgr10_head_balance_5090.log`。
SHA256：`2d18b056f25586cc850ac25c8d57b46d9b012f0190d3dd6364d4525e6eed52db`。
只用 `imgr10_t10_head_balance_off_seed1993_5090_20260927_094456_full` 的前三任务，
不是其head_balance候选，也不是3090结果或80%训练的holdout结果。

- CNN曲线：[96.81, 92.28, 89.93]；三阶段Average=93.0067。
- Task1 Last=92.28，Old=94.63，New=89.72，Forgetting=2.1800。
- Task2 Last=89.93，Old=91.22，New=87.13，Forgetting=2.1350。
- 历史参考与本次提交不同，不声称严格同提交重跑。运行后先核对有效配置、Task0结果及GPU/软件环境；
  若起点不匹配，应报告混杂，不将差值归因为S学习率。

重点同时报告Task1和Task2的Old/New、Average、Last、遗忘，不能只看某一项或最佳任务。
只有New不损失且Old/Total出现净收益，才值得另行确认完整T10；
若只提高Old、继续压低New，则仍未解决目标问题。T3成绩不能当完整T10结论。
这是探索性官方测试消融；若据此继续选配方，应如实披露，不包装成独立无偏测试验证。

## 运行

```bash
# 可选通路短测：1轮/Task0–1，成绩不作对比
bash scripts/9_28_imgr10_slora_lr_t3_5090.sh --smoke
# 正式仅1次候选，无自动短测或基线重跑
lrun scripts/9_28_imgr10_slora_lr_t3_5090.sh ./logs/9_28_imgr10_slora_lr_t3_5090.log
```

配置：`scripts/sweeps/imgr10_slora_lr_t3_5090.json`；
逐次日志：`logs/shell_logs/imgr10_slora_lr_t3_5090/<timestamp>/slr0p5.log`。
外层记录commit/工作区状态，内层保存完整命令和退出码；trainer记录最终配置。
检查Task0初始lrs=[0.02,0.02]，Task1/2=[0.02,0.02,0.01]（LoRA/classifier/S）。
原匹配基线09:45至10:36完成前三任务，约51分钟；共享GPU占用变化，预算约1小时而非保证时长。

## 验证与待回填

本地专项测试验证默认/Task0不变、S/P独立分组、无重复参数、SGD一步比例和余弦比例，
以及正式队列仅1次、全量训练、anchor5；无训练中验证断言。
专项及相邻脚本回归16项通过；Python编译、Bash语法、dry-run、JSON矩阵展开通过。
独立代码审查通过，未发现阻断问题；确认实际增量训练仅缩放当前S-B，历史参数保持冻结。
全量轻量回归269项通过，另2个测试模块（test_ca_diagnostics、test_task0_margin_screen）
因本机缺少easydict无法导入；不声称全部通过，也未改无关依赖。
本地无CUDA，GPU短测/正式训练尚未执行，未登录或启动服务器。
运行后回填原日志hash、退出码、实际配置、Task0一致性和全部上述指标，再作保留决策。
