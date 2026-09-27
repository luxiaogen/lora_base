# 9/28 两机夜间实验：新分类头起点与 Task0 anchor

状态：实现与 CPU 验证；没有在服务器启动，尚无 CUDA/性能结果。
原项目 `codex/mask-budget-comparison-20260924`；配置文件和本机 `data_path` 不修改。

## 固定配方

ImageNet-R、seed1993、T10 类别顺序/每任务20类；Task0/后续20轮、CA5、math-SDPA。
原 layer 双门、P 可塑区、训练正则0.01、冻结 P-A、P LR0.02。
独立 anchor 仅 Task0：3090=10，5090=5。新增蒸馏、旧类竞争、B方向修正均关闭。
短程为 Task0–2，每类训练数据确定性预留20%；所有 train-source 请求（含本次
分类头特征、W0统计、CA）排除 holdout。旧类 holdout 仅评估，不优化或作teacher输入。
完整 T10 重新从头使用全量训练数据，不从短程模型接着训练；测试仅报告。

## 组别

| 名称 | 初始化 | 额外分类头训练 | 改变的变量 |
|---|---|---|---|
| control / G0 | 原随机 | 无 | 短程对照 |
| prototype / G1 | 类别原型方向，保留每行原范数 | 无 | 初始化方向 |
| warm / G2 | 原随机 | 联合训练前5轮 | 先适应分类头 |
| proto_warm / G3 | 同G1 | 联合训练前5轮 | 两因素组合 |
| head_post / G4 | 原随机 | 合并后、CA前5轮 | G2等额更新步数对照 |
| proto_head_post | 同G1 | 合并后、CA前5轮 | G3对应的训练量对照 |

G1–G4 不改变 Task0。新增三个开关默认完全关闭：
`head_start_init=random`、`head_start_epochs=0`、`head_start_stage=pre`。
原型来自当前任务**训练图**的中心裁剪特征，特征模型为之前任务合并后的模型，不是 W_pre。
预训练和后训练都只优化当前 head：余弦分类+原CosFace（scale20/margin0.1），SGD
LR0.02、momentum0.9、5轮余弦调度、batch48、无weight decay。特征缓存仅本阶段驻留，
不保存旧图片、不增推理前向。后训练重新提取当前模型特征；它与预训练不是同一特征矩阵，
只是相同训练样本和更新步数，所以该对照不能完全隔离所有表征因素。
额外步骤用局部随机生成器，不改变主训练 RNG；正常联合训练使用全新 optimizer。
日志 `HeadStart` 记录时点、特征任务、样本数、额外epoch/步数和耗时。

3090另做 anchor=0/2.5/5/10/20 的独立短程消融；关闭全部新head改动，只改变
`dual_mask_anchor_reg_weight`。0表示独立anchor损失为0，不关闭双门或掩码正则。
10为对照复用；既往anchor5全数据实验不能替代这次holdout对照。

## 运行顺序和自动续跑

- 3090复用归档的 anchor10 短程control，顺序跑 G1、G2、G3、anchor0/2.5/5/20：7次T3。
- 5090跑本机 anchor5 的短程control、G1、G2、G3：4次T3。
- head候选通过即排完整T10；anchor最多选通过者中平均验证Total增益最大的一组。
- G2或G3通过，各自再排匹配初始化的等额分类头更新对照。
- 最多3090：7次T3 + 6次T10；5090：4次T3 + 5次T10。明确无净收益则不续跑。
- 不重复已有完整T10基线，不把head候选与筛出的新anchor组合，避免一次改变两项。

推进标准（看结果前固定）：Task1/2等权平均验证Total增益>=0.20pp，平均Old/New均不降，
任一任务Old/New不下降超过0.30pp。head候选还要求Task0验证准确率一致，所有候选要求
验证样本ID/标签哈希和类别范围一致。anchor改变Task0，故不要求它的Task0准确率一致。
这些是实验选择规则，不是训练路径的assert或单元测试；不以测试集最高分决定续跑。
验证集筛出的最优anchor只是这五个值、这个seed、这个协议下的候选，不是全局最优。

3090参考：`scripts/sweeps/imgr10_holdout_reference_3090.json`。
原日志 `9_27_imgr10_distill_holdout_t3_3090.log`，commit c416e0d，weight0 run，
CPU侧提取的原始指标/配置/原日志SHA256均保留。没有把原始大日志或图片提交到Git。
代码/数据/依赖或GPU设置不同，请加 `--fresh-control` 重新做一次短程control；
样本ID哈希不等于图片内容或完整环境哈希，不能仅凭它证明跨环境可比。
默认重复使用原3090数据/环境，候选Task0不匹配时禁止自动推进，而非静默换参考。

## 命令

```bash
# 查看所有计划命令，不执行、不创建日志
bash scripts/9_28_imgr10_head_start_anchor_3090.sh --dry-run
bash scripts/9_28_imgr10_head_start_5090.sh --dry-run

# 可选GPU通路检查：每组Task0–1各1轮，额外head1轮；不排名，不续T10
bash scripts/9_28_imgr10_head_start_anchor_3090.sh --smoke
bash scripts/9_28_imgr10_head_start_5090.sh --smoke

# 夜间队列（不要两条都放同一台机器）
lrun scripts/9_28_imgr10_head_start_anchor_3090.sh ./logs/9_28_imgr10_head_start_anchor_3090.log
lrun scripts/9_28_imgr10_head_start_5090.sh ./logs/9_28_imgr10_head_start_5090.log

# 可选只做短程；手工续跑已选候选（不会重跑control）
bash scripts/9_28_imgr10_head_start_anchor_3090.sh --screen-only
bash scripts/9_28_imgr10_head_start_anchor_3090.sh --full prototype anchor5
```

参数开关不自动保存checkpoint。每次调用创建独立带时间戳的日志目录：
`logs/shell_logs/imgr10_head_start_anchor_<machine>/<timestamp>/`，含每组原始日志、
`manifest.json`（提交/工作区状态/规格）、`holdout.csv`、`decisions.json`。
一次训练失败会保留退出码并继续其他组；失败候选/control不会触发完整实验。
单元测试不串进夜间脚本。CPU测试/语法检查不能代替GPU短测或证明性能提升。

预算：3090短程约18–25分钟/组、完整约80–100分钟/组再加head准备开销，
最坏全部推进可超过10小时。5090共享GPU不承诺固定结束时间，队列串行运行，不干预别人。
可以用现有 `lwait PID "lrun ..."` 等当前训练结束后再启动。
