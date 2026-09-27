# 2026-09-28 新分类头准备 / anchor消融：待运行

本记录是预注册安排，不是性能结果。实现见
[运行说明](../../scripts/README_head_start_night.md)。

## 前置结论与不重复事项

- 已完成蒸馏holdout：`/Users/luxiaogen/Desktop/loda_logs/9-27/9_27_imgr10_distill_holdout_t3_3090.log`，
  commit c416e0d；三个Task0–2均完成、exit0。Task1–2验证均值：

| 权重 | Total | Old | New |
|---|---:|---:|---:|
| 0 | 89.9451 | 91.7260 | 87.7925 |
| 0.25 | 89.5556 | 91.7681 | 86.7963 |
| 1 | 89.3717 | 91.7221 | 86.5165 |

两候选未达预定标准，停止该蒸馏权重扫描。将weight0作为同机、同配方的短程参考，
指标/配置/原日志SHA256在 `scripts/sweeps/imgr10_holdout_reference_3090.json`。
- 历史anchor5做过全数据T3和部分T10，不能当作本次训练holdout结果。
- 新实验不再解冻P-A、不改变预算/粒度、不叠加旧类竞争，不重复完整基线。

## 待检验假设

新随机分类头尚未适应已有特征时即联合训练LoRA，可能产生不必要的特征改动。
这只是候选解释；不能把初始化改动有收益直接写成已证明“早期漂移是瓶颈”。

两机都比较随机/原型初始化 × 无/有head预训练；有收益再加等额后训练对照。
Task0不改。3090另测Task0 anchor0/2.5/5/10/20；其他机制独立固定。
筛选仅用训练holdout；选择标准、续跑规则、顺序、最大次数见运行说明。
3090固定anchor10，5090固定anchor5；不能跨机相减，更不是多seed证据。

## 实际运行后补齐

提交hash、GPU/依赖与占用、manifest路径、完整原日志路径、每组退出码/完成任务数、
holdout.csv、decisions.json、全T10 Average/Last/Old/New/Forgetting及额外耗时。
只有完整T10相对匹配本机基线出现净收益才考虑保留；当前未启动服务器。

## 本地验证

- 新增head函数/调用路径、队列选择、旧holdout与蒸馏回归专项：27项通过。
- Python编译、两台脚本Bash语法与dry-run通过；JSON矩阵展开通过。
- 独立代码审查已核对holdout隔离、pre/post合并位置、RNG、初始化范数和新optimizer。
  据审查补齐了两种初始化各自匹配的训练量对照。
- 全量轻量回归中，`test_ca_diagnostics`、`test_task0_margin_screen`因本机缺少
  `easydict`无法导入；不声称全套通过。未改这些无关依赖和测试。
- 无本地CUDA/服务器GPU短测，耗时和准确率均待实际运行确认。
