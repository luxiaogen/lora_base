# 有限步幅 / 分支分工参照与合法信号

## 不是严格的性能上界

前一批真实旧梯度投影（339fed2）T3没有净收益。新实验不再重复那条投影：
用有限候选的实际前向，检验是否存在更好的S/P一步分工。
它是“受限特权参照”，而不是保证高于基线的Oracle上界。
候选集合包含原SGD，所以在本步选择批次上有一个可行原点；
这不保证后续训练轨迹、独立探针或测试准确率更好。

| 机器 | 范围 | 候选 | 公平性边界 |
| --- | --- | --- | --- |
| 3090 | 只改P当前B步幅；S保持原SGD | P×1、0、0.5、1.5、2 | 有意改变P步长，不是等干预强度对照 |
| 5090 | 改S/P当前B步子的分工 | S/P原步；0/1、0.5/1.5、1.5/0.5、1/0后归一化 | 匹配联合B位移范数，不匹配门控后QKV更新范数 |

Task1以后A仍冻结；Task0不启用。先执行原SGD，再在相同的分类头与动量状态下，
临时换当前B位移并实际前向。采用的候选直接成为本步B；动量仍为原SGD动量。
这是后SGD步子过滤，不是另一种动量优化器。两分支不改变原保护/可塑/冲突门。
临时候选会通过原layer动态门重新计算；最终merge仍走原规则。
联合范数归一化在非零能量上直接算比值；某个候选方向完全为零时回退原步。
日志同时记录目标与实际写入的B位移范数，保留浮点舍入偏差，不能只报理论倍率。

## 四种模式与数据边界

| branch_choice_mode | 采用哪一步 | 是否读旧训练图片 |
| --- | --- | --- |
| audit | 永远原SGD；候选只读记录 | 是，仅特权诊断，不决定模型更新 |
| oracle | 在旧训练CE不高于原步的候选中，选新训练CE最低者 | 是，特权参照，不能当无回放方法 |
| logit | 用当前新图片上的旧头分数变化作为风险，选新CE最低者 | 否 |
| feature | 用当前新图片上的特征变化作为风险，选新CE最低者 | 否 |

选择用的new CE是全部已见类别上的scaled CE，正确标签来自当前训练批次；
它不是原训练CosFace损失的替换，只用来比较临时候选。分类头保持共同的post-SGD状态。
原SGD训练目标、正则和anchor不变。测试数据仅按原协议报告，不选候选、不晋级、不拟合信号。

合法信号的参考是本步B更新前的特征，不是额外教师模型：

- logit：同一新图片，旧头余弦分数前后变化的均方值；不降低旧分数、不反传此项。
- feature：归一化CLS特征前后差的平方范数均值×0.5。

P实验只把P回到步前，S固定post-SGD；S/P实验两者都回到步前。
这两个量只能反映当前新输入的变化，不能默认代表真实旧知识受损。
Oracle风险容差为CE的1e-6；两个小量代理使用原风险的0.1%+1e-14数值容差。
改善new CE不足1e-7时保持原步；并列也保持原步。没有新增可调损失系数。

特权数据池复用原Oracle划分：每个旧类最多4张selector、4张互斥probe；
每个新类最多4张probe。只读训练来源，确定性预处理，独立loader generator。
探针不用于选择，也不是未见holdout。合法模式不构造这些loader。
旧图片/路径仅在任务期间临时使用，不保存旧图片、稠密特征、历史增量或checkpoint。

## 两台队列

共同配方：ImageNet-R T10，seed1993；anchor2.5；Task0/增量20epoch；CA5；math-SDPA；
原layer粒度、自适应设置、P学习率0.02。其余近期候选关闭。JSON路径/设备/预训练文件读取本机配置。

| 顺序 | 3090，6小时启动预算 | 5090，8小时启动预算 |
| --- | --- | --- |
| 0 | audit/oracle/logit/feature各一组2任务1轮短测 | 相同 |
| 1 | audit_t3：只读反事实，不改原模型 | audit_t10：只读反事实兼同提交完整对照 |
| 2 | oracle_t10 | oracle_t10 |
| 3 | logit_t10 | logit_t10 |
| 4 | feature_t10 | feature_t10 |
| 余量 | oracle/logit/feature_dense_t3：从每5步比较改为每步比较 | 相同 |

默认间隔5步；前三任务稠密检查是频率对照，不是假装额外seed。
5090完整anchor2.5对照此前缺失，因此audit走T10。3090正式候选复用本机已有T10，
要核对Task0与完整配方；历史提交不同，微小差异不能作强因果结论。
不因测试成绩差自动停止后续组。所有实验优先级预登记，预算不足的组打印budget_skipped与续跑命令。

有多个临时候选前向，不能用普通T10的80分钟直接报这批耗时。先用第一组实际耗时更新估计，
启动下一组前留15%余量；不杀正在训练的进程，也不保证恰好6/8小时结束或所有备选都跑完。
预算时间从队列启动计算，包含短测；lwait等待当前队列的时间不计入新队列。
预算队列需要逐次实际耗时，因此用JSON展开Python启动命令，不用无条件串行的静态sweep脚本。

## 运行（在原项目）

```fish
git switch codex/mask-budget-comparison-20260924 && git pull --ff-only origin codex/mask-budget-comparison-20260924
git rev-parse --short HEAD
```

本地修改冲突或分支占用时先停止更新，不覆盖data_path或本地代码。

3090：

```fish
lrun scripts/10_01_imgr10_branch_choice_3090.sh ./logs/10_01_imgr10_branch_choice_3090.log
```

若当前外层队列还在跑，先填真实外层PID，代码拉取也等它结束后：

```fish
lwait <外层队列PID> "git pull --ff-only origin codex/mask-budget-comparison-20260924 && lrun scripts/10_01_imgr10_branch_choice_3090.sh ./logs/10_01_imgr10_branch_choice_3090.log"
```

5090：

```fish
lrun scripts/10_01_imgr10_branch_choice_5090.sh ./logs/10_01_imgr10_branch_choice_5090.log
```

以上lrun/lwait用你已有Fish函数，Bash脚本本身可用`bash scripts/...sh`。
预览用`bash scripts/10_01_imgr10_branch_choice_3090.sh --dry-run`；只短测用`--smoke`。
追加指定组例：`bash scripts/10_01_imgr10_branch_choice_5090.sh --only feature_t10 --hours 3`。
我只修改本地并推送GitHub，不登录/拉取/启动服务器。

## 输出与判断

每组日志、manifest.json、queue.json在各机
`logs/shell_logs/imgr10_branch_choice_<machine>/<时间戳>/`。

每个采样步记录5个候选、选择结果、new CE、两个代理、B步长；特权模式额外记录old CE。
第1/10/20轮首采样步有独立训练探针，audit报告的是假想Oracle，不写入模型。
只记标量；辅助评估保留随机状态、模型模式和优化器状态。

队列自动生成candidate_steps.csv与signal_summary.json：每条轨迹、每个任务分别计算代理AUC和误报/漏报。
这里预测的是“采样旧训练CE是否更差”，不是测试遗忘。审计没有改变模型，最适合检查未干预轨迹的信号；
Oracle轨迹已被选择规则影响，不能与审计混池。原始CSV可供后续图形分析。
同一步的4个非原候选互相关联，AUC/计数是描述性诊断，不是独立样本显著性检验。

判断分三步：候选确实采用 → 特权参照独立探针/完整结果是否有收益 → 合法信号/候选是否能逼近。
若参照有效但代理不相关，只能说旧监督有价值；若代理相关但合法T10无净收益，也不能说已解决问题。
有限参照失败仅否定这套候选集合/选择规则，不等于所有Oracle无效。

本地CPU测试、Shell语法、dry-run与代码审查不替代服务器CUDA短测或性能证据。
