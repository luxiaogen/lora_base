# P冲突掩码后段固定：3090 Task0–2

状态：候选实现与短程筛选，不是已经证实的提点方法。

## 假设

P更新变化导致冲突位置重新选择，可能使后期优化面对变化的约束。第5轮末捕获实际P冲突掩码，后15轮沿用它；仍同时训练S/P的B与当前分类头。冻结位置不是冻结参数，也不是削弱设定的门控强度。

历史末5轮平均无净收益，S5/P15完整T10低于基线，因此本候选显式关闭两者。不调整CA、分类头损失、初始化、rank或学习率。历史oracle不是该假设成立的证据。

## 精确配方

- 原项目、原分支；仅3090运行，ImageNet-R seed1993。
- `scripts/sweeps/imgr10_p_mask_freeze_3090.json`记录完整覆盖项，data_path/device仍读机器JSON。
- anchor2.5、Task0/增量20轮、CA5、math-SDPA、原layer双门、原自适应设置、P private_conflict_mode=global。
- `p_conflict_freeze_epoch=5`，默认0关闭。Task0不执行。每增量任务清空上个任务缓存；第5轮结束、最后一次优化器更新后重新计算该权重状态下实际前向mask，供第6–20轮及最终merge使用。
- 第5轮后只替换P冲突坐标选择，不冻结S/P参数；正则评分仍按原公式动态计算，使用的有效P更新遵循固定门。
- mask缓存为当前任务bool缓冲，不进入state_dict；merge后清空。正式推理无额外分支、无任务ID、无额外前向。
- `save_task_weights=false`、`late_weight_average_epochs=0`、`sp_staged_s_epochs=0`，不改任何默认JSON。
- 本轮支持/验证范围是上述layer/global-P配方，不与global预算或plastic_norm_matched候选叠加。

## 观测与归因边界

每层每轮末 `PConflictFreeze` 记录实际/动态候选密度、相对上轮实际mask的切换比例和Jaccard、实际mask与当前动态候选Jaccard、二者对应移除更新的范数/可塑区更新范数。动态候选只用于日志，不参与更新。

切换率是epoch-end采样，不是逐优化步切换统计。固定前后坐标数在冻结时刻对齐，但此后原自适应规则可能改变动态候选数量；固定坐标不保证同等抑制能量。若New改善但实际抑制量下降、Old受损，不能宣称优化稳定性改善。

## 运行及对照

脚本 `scripts/9_29_imgr10_p_mask_freeze_3090.sh` 默认先短测（Task0 1轮、Task1 2轮、第1轮末固定、CA1），成功后运行唯一正式Task0–2候选。`--t3`跳过已完成短测。无准确率断言、无测试集择优开关。

比较同机基线 `/Users/luxiaogen/Desktop/loda_logs/9-28/9_28_imgr10_anchor2p5_save_t10_3090.log` 前三任务：97.10/92.73/90.24，Average93.3567；Task2 Old91.60、New87.29、Forgetting2.155。对照为历史运行、保存开关/代码提交不同，不能把微小差异当显著提升。原始T3日志93.x不与完整T10的87.x比较。

本次训练预计约22–30分钟加约3分钟短测，参考9/29 late_average实际21.81分钟。测试准确率用于描述性报告，后续优化筛选需要保留独立验证与未用于反复调参的评估。

只有Old基本保持且Average/Last净改善，才考虑完整T10；不自动追加实验，不占用5090。
