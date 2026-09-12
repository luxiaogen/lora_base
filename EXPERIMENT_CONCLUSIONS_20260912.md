# 2026-09-12 实验结论与主分支冻结记录

## 主分支与后续实验规则

用户决定将本地及 GitHub main 精确恢复到 `102b65c8e6aa7c401868b029c7e2509e64ffe8a0`。
该提交日期为 2026-09-09，标题为 `Add read-only CA diagnostics and fixed64 overnight sweep`。
本文保存在 `codex/task-score-routing` 实验分支，不写入被冻结的 main。

回退前引用：

- GitHub main：`207f00f664e84bcdbd0de615aaee58fc31516e4d`。
- 本地 main：`78b0da88032852330f4951a588b7401b7fc5f756`。
- 当前诊断实现：`cb44884f9b69596011c2f87fedda6ba625c84fb4`。
- 远端 main 备份：`codex/archive-main-20260912-207f00f`。
- 本地 main 备份：`codex/archive-local-main-20260912-78b0da8`。

保留 `codex/task-score-routing`、`codex/boundary-real-current`、`codex/overnight-9-12` 及备份分支。失败实验不合并 main。新候选从 main 新建 `codex/<experiment>`，验证有效后仅合并需要的改动；不要将包含多个失败方案的历史分支整体合入。

## 代码版本不等于实验协议

102b65c 的 `exps/dlora/imgr10.json` 默认值：CA10、reg_weight=0.1、conflict_reg_enabled=true。
最近 ImageNet-R 筛选实验的实际覆盖为：

```text
seed=1993，T=10，init_cls=20，increment=20，rank=64
init_epoch=20，epochs=20，ca=true，ca_epochs=5
dual_mask_task0_gate_mode=unmasked
dual_mask_anchor_reg_enabled=true
dual_mask_anchor_reg_weight=10
dual_mask_anchor_reg_task0_only=true
dual_mask_conflict_energy_adaptive=true
dual_mask_conflict_energy_ratio_floor=true
dual_mask_conflict_ratio=0.1
dual_mask_conflict_strength=0.5
dual_mask_conflict_old_overlap_adaptive=true
dual_mask_private_conflict_mode=global
dual_mask_conflict_merge_mode=suppress
dual_mask_conflict_reg_enabled=false
dual_mask_reg_weight=0.01
```

数据路径使用每台服务器实际配置。先备份服务器本地配置，回退后仅恢复所需本地路径，不要整体恢复旧实验代码或整份 JSON。此回退不承诺重现某个单次 87.1 的结果。

## 结果（pp 为百分点）

以下数据来自实际日志。Task0 checkpoint 恢复运行可能被通用解析器标成 2/10 或 9/10，因为只记录续跑任务；应结合恢复记录和完整精度曲线判断。前三任务筛选与完整十任务结果不混算。

### 1. 分类头梯度隔离：当前设置不合并

Task0 checkpoint 配对、跑至 Task2：

|机器/候选|Last 基线→候选|Old 基线→候选|New 基线→候选|Forgetting 基线→候选|
|---|---:|---:|---:|---:|
|3090 detached 当前样本分类头损失|89.67→88.84|91.07→88.87|86.63→88.78|2.305→5.335|
|5090 detached 分类头损失＋伪特征 replay|90.14→89.21|91.52→89.48|87.13→88.61|1.845→4.720|

新任务改善伴随旧任务损失；隔离 LoRA 梯度未解决这个取舍。

### 2. Boundary task bias：收益小且新旧取舍明显

3090、三 seed、每个 seed 使用同一 Task0 checkpoint，完整 T=10：

|Seed|Average 基线→候选|Last 基线→候选|New 基线→候选|
|---|---:|---:|---:|
|1993|87.058→87.092|82.23→82.45|85.49→83.94|
|1996|86.322→86.401|82.03→82.17|87.34→85.53|
|1997|86.389→86.433|82.30→82.52|89.80→87.93|

Average 平均增益约 +0.052pp，New 降低约 1.55–1.87pp。旧类改善不能单独证明整体方法更好。

真实当前任务特征＋伪旧任务版本，在5090 seed1993 Task0–2配对中：Last 89.72→89.78，Old 90.99→90.76，New 86.96→87.62，Forgetting 2.235→2.615。未满足保住 Old/Forgetting 的预设筛选要求。

### 3. 上一任务功能保持 teacher：遗忘下降但性能下降

5090、三 seed、完整 T=10 配对：

|Seed|Average 基线→候选|Last 基线→候选|Forgetting 基线→候选|
|---|---:|---:|---:|
|1993|87.111→86.609|82.43→82.18|5.4478→3.9778|
|1996|86.389→85.975|82.25→81.72|6.9133→5.6856|
|1997|86.341→85.821|82.50→82.32|6.6811→4.8367|

不能将 Forgetting 下降直接当作成功：新任务起始精度降低也能使遗忘变小。必须同时检查 acquisition accuracy、最终 Old、New、Average。

### 4. 头内置信度路由：同模型配对诊断失败

5090 seed1993 Task2，同模型、同 logits：

|规则|类别准确率|Old|New|任务识别率|
|---|---:|---:|---:|---:|
|global|90.04|91.22|87.46|92.73|
|centered_max|89.93|91.37|86.80|92.58|
|top1_top2|86.14|87.28|83.66|87.86|

只支持停止已测规则，不能据此否定所有任务路由。原始任务最大分数再选类与全局 argmax 数学等价，不是新算法。

### 5. NCM 第二证据：当前低-gap替换无收益

使用当前模型特征及保存的训练类别均值；不是 W_pre-only NCM。
5090 seed1993 Task2，1927个样本：

- global 任务识别92.84%，NCM 91.23%。
- 两者都对1729；NCM能救回29；NCM新增错误60；两者都错109。
- oracle_union=94.34%是借助真实标签挑选证据的上限，不是可部署性能。
- 最低任务-gap约10%：193个样本，捕获49.28%的global任务错误。
- 在该子集切换到NCM，救回24、新增46，整体任务识别92.84→91.70%，下降1.14pp。
- Task1同样下降0.38pp。

这些是任务识别率，不是 CNN 类别准确率；实际 CNN 仍为 global，Task2=89.93%。最低10%由整个测试集排序得到，仅作离线诊断，不属于已验证的逐样本推理规则。

旧原型漂移是可能解释，当前交集统计没有证明其因果作用。NCM与分类头共享特征，不能称统计独立证据。全局 recoverable>introduced 也不是所有条件融合的必要条件；本次停止依据是已测子集净收益为负。

## 可以保留的发现及需要收紧的推论

1. CA 在最近三次 Task2 的 Total 分别提高0.73、0.57、0.36pp，Old均未下降；但不能推广为每个阶段都不伤旧类（Task1有旧类下降）。
2. 最新 Task2 global=89.93%、oracle task ID=94.76%，证明跨任务竞争造成错误；不能单凭差距断定根因只有分类头分数不可比。
3. 原始最大头分数对大部分样本有效；pooled AUC不是同一样本正确头战胜最强错误头的概率。
4. 低任务-gap能集中发现风险，但替代决策未成功。参数 conflict_ratio 与样本歧义比例是不同概念，复用0.1只是诊断选择，没有理论等价关系。
5. 最近三次 Task0/Task2 分别为96.81/90.04、96.23/89.98、96.52/89.93。尚不能证明Task0越高最终一定越好；这些运行也不是隔离所有随机因素的因果实验。
6. 后续仅训练P、Task0保留S，是尚未验证的结构消融候选，不应提前合并。P更新同样进入累积主干，不保证旧功能冻结；若减少S，参数量也是实验因素。

## 下一阶段验收约定

- 新候选使用独立实验分支，同一机器、同一数据划分、同一有效配置，并共享Task0 checkpoint（需要时从实验分支选择性引入该功能）。
- 短程筛选同时记录Average、Last、Old、New、任务识别及学习时精度；不只优化Forgetting。
- 有正向信号再跑完整T=10及配对多seed；短程、单seed和跨机器未配对数据不能当作最终改进结论。
- 测试集诊断不用于训练、更新统计量或选择部署阈值；候选规则/阈值需在训练侧固定后另行验证。
- 合并需要性能证据及回归检查，保留有效命令、commit和原始日志。

## 本机日志索引（日志本体未上传）

- `/Users/luxiaogen/Desktop/loda_logs/9-11/9_11_head_isolation_pair.log`
- `/Users/luxiaogen/Desktop/loda_logs/9-11/5090/9_11_head_isolation_pair.log`
- `/Users/luxiaogen/Desktop/loda_logs/9-12/9_12_boundary_calibration_3090_overnight.log`
- `/Users/luxiaogen/Desktop/loda_logs/9-12/9_12_previous_function_5090_overnight.log`
- `/Users/luxiaogen/Desktop/loda_logs/9-12/5090/9_12_boundary_real_current_screen.log`
- `/Users/luxiaogen/.codex/attachments/cb623be0-5de6-42a1-b292-809aae14e917/pasted-text.txt`
- `/Users/luxiaogen/.codex/attachments/ac4e03ee-4acb-45b3-9e24-f60a49c77102/pasted-text.txt`
- `/Users/luxiaogen/.codex/attachments/3e73e33d-7886-43ac-9a7d-b265c3e4656c/pasted-text.txt`
