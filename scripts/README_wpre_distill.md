# W_pre 判对样本上的 S 分支特征保持候选

不是推理时选择专家，也不是测试标签 Oracle。本次实际改变 Task1–9 的训练梯度，正式推理仍用原 DualMask 一次前向。

## 唯一新增机制

1. 复用每个任务开始时已有的 W_pre 训练特征提取，累计 `FrozenReadout` 的 Gram 和类别特征和；原图和逐样本特征不保留。
2. Ridge 正则 alpha 固定为 1，不由测试成绩选择；分类范围为全部已见类。
3. 每个训练 minibatch 用同一份增强图片，先无梯度提取原始 AugReg ViT 特征，再执行原学生前向。
4. 用**当前训练标签**判断教师是否判对；判对样本参与余弦特征距离的平均，判错样本不增加该项，权重固定 1。
5. 单独求辅助损失对当前 S.B 的梯度，与原梯度相加；不向 P、A、分类头直接添加辅助梯度。

普通分类损失仍训练 S/P/分类头。特征来自完整学生模型，**不是独立 S 网络**；后续 P 训练轨迹可能因 S 改变而间接改变。
教师同样在当前训练图片上拟合，判对筛选是训练内监督，不能称作独立验证或证明教师可靠。
该损失没有直接约束旧图片，不保证保住 Old；要用完整 T10 检验净收益。

Task0 没有新教师前向、辅助损失或梯度；只额外累计确定性的 CPU 统计。辅助前向恢复训练模式以及 Python/NumPy/Torch 随机状态。
不改 S/P 双门、掩码选择、正则、A 冻结、merge、CA、推理。默认 `wpre_distill_weight=0` 完全关闭新方法。

## 两机分配

两机各一个相同候选：ImageNet-R、seed1993、anchor2.5、20 epochs、CA5、math-SDPA，完整训练集，Task0–9。
无训练留出、旧图片回放、真实任务 ID 推理、测试调权或权重保存；每台与本机已有**全训练集**基线比较。
最近 Ridge 融合的 5%/10% 留出配方不是本次性能对照。两机不能横比绝对准确率。

3090：

```bash
bash scripts/10_01_imgr10_wpre_distill_3090.sh
```

5090：

```bash
bash scripts/10_01_imgr10_wpre_distill_5090.sh
```

默认先 2 任务 × 1 epoch GPU 短测；退出 0 后进入完整 T10。短测成绩不用于挑参数。
如只需检查命令，末尾加 `dry-run`；单独短测加 `smoke`，只跑正式实验加 `t10`。
数据路径仍取各服务器本机 `exps/dlora/imgr10.json`，入口不覆盖。

## 记录与判据

日志目录 `logs/shell_logs/imgr10_wpre_distill_<machine>/<timestamp>/{smoke,t10}/`，各含 `training.log` 和 `run.json`。
`run.json` 固定提交、实际配置、命令、关键源文件哈希。
`WpreDistillTeacher` 记录当前/累计训练样本数、类别范围、统计量大小；768维 × 200类时约2.84 MiB。
`wpre_selected_ratio`、`wpre_feature_loss` 每 epoch 汇总；少量步骤的 `WpreDistillGrad` 记录实际 S 辅助梯度与 CE 梯度范数、夹角。
训练比原基线多一次无梯度教师前向和 S 辅助反传；不能承诺原训练耗时。

只有 Average/Last 有净提升且 Old/遗忘没有明显受损，才支持继续验证。
教师选中率与辅助梯度非零只证明代码生效，不证明性能有效。
