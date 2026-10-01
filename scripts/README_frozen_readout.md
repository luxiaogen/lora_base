# 固定 W_pre：NCM 与增量岭回归

实验编号：DM-B-261001-006；ImageNet-R T10，seed1993。**不重新训练 DualMask。**

本轮回答：此前预训练 NCM 太弱，是否只是“类别均值 + 余弦”没有充分读出原始 AugReg ViT-B/16 的知识？
它不是已验证的路由方法，也不是完整 RanPAC 复现。

## 两机分配

| 机器 | 实验 | 复用的原模型预测 | Task1+外层校准排除 |
| --- | --- | --- | --- |
| 3090 | 同一固定特征上的 NCM / Ridge，完整 T10 | 本机刚完成的 expert_legal T10 | 每类约5% |
| 5090 | 同上，独立在本机配对 | 本机刚完成的 expert_legal T10 | 每类约10% |

两机是相同 seed 的本机复核，不是多 seed；外层排除比例不同，不横比绝对准确率。
两个读出再共同使用剩余数据中每类约80%拟合（沿用原 W_pre NCM 的内部留出规则）。
Task0 不做外层排除。**所有拟合只读取当时的当前任务训练图片；不回放旧图片。**

## 更新规则与空间

固定原始 W_pre，不训练 A/B、骨干或 CA。原始768维 CLS 特征，不加随机投影。
两种读出使用完全相同的拟合样本与原始特征：

```text
G ← G + X_currentᵀ X_current
Q ← Q + X_currentᵀ Y_current
NCM：归一化 Q 的每个类别列，做余弦分类
Ridge：W = solve(G + λ I, Q)，做线性分类
λ = α × trace(G) / 768
```

α 候选预声明为 0.0001 / 0.001 / 0.01 / 0.1 / 1。
仅 Task0 训练留出集选 α，准确率并列取列表首个；选定后写入 `task0_alpha.json`，**再首次读取测试图片**。
后续固定 α，λ 按上式随累计统计变化；不是在每任务重新调参。
留出样本不回填拟合统计。测试标签和真实任务 ID 不进入预测、选 α 或更新。

G/Q 的长期统计仅2,973,696字节（约2.84MiB），不随图片数增长。
当前任务训练特征计算后即释放，不保存图片、训练特征或模型权重。
为了每张测试图片只提取一次，诊断脚本在 RAM 暂存已见测试特征，最终约17.6MiB；
它们只用于报告，从不拟合读出，且不写磁盘。磁盘只留小型 JSON / 预测 CSV。

## 运行

在各机器现有项目根目录、原有环境执行，不创建新项目，不修改本机 JSON 路径。

3090：

```fish
lrun scripts/10_01_imgr10_frozen_readout_3090.sh ./logs/10_01_imgr10_frozen_readout_3090.log
```

5090：

```fish
lrun scripts/10_01_imgr10_frozen_readout_5090.sh ./logs/10_01_imgr10_frozen_readout_5090.log
```

预览配置（不加载模型、数据、CUDA）：

```bash
bash scripts/10_01_imgr10_frozen_readout_3090.sh --dry-run
bash scripts/10_01_imgr10_frozen_readout_5090.sh --dry-run
```

可选 `--smoke` 只做 Task0–1 特征读出，不是1轮训练；默认直接完整 T10，避免自动重复提取。
这里没有20轮反向训练，耗时取决于一次图片编码与 CPU 求解；首次 CUDA 耗时待实测。

## 缓存和结果

默认缓存已按用户提供的两份日志中的目录固定，不搜索“最新结果”：

```text
3090 logs/shell_logs/imgr10_expert_legal_3090/20261001_150130_524490/t10/diagnostics
5090 logs/shell_logs/imgr10_expert_legal_5090/20261001_150145_686685/t10/diagnostics
```

若移动过该目录，可以传 `--reference-dir /实际/t10/diagnostics`（它的上一级需有 `run.json`）。
读取参考有效配置，只用本机 JSON 覆盖 data_path/device；参考 seed、类别协议和 math-SDPA 不匹配则不配对。
每任务核对缓存索引、标签，并报告新算 NCM 与原缓存 NCM 的预测一致率。
缺缓存或出现任何 NCM 预测差异时，**继续独立读出，不终止实验，但该任务不生成严格配对的互补上界**；
旧缓存没有图片内容/原骨干哈希，不能据此宣称跨时间逐位复现。不要人为设“可接受差距”来放行结果。

输出：`logs/frozen_readout/{3090,5090}/时间戳/`。

- `manifest.json`：提交、源代码哈希、有效配置、类别顺序、骨干参数哈希、路径列表哈希、版本。
- `task0_alpha.json`：唯一参数筛选过程。
- `task_XX.json` / `task_XX_predictions.csv`：全部已见类别成绩，Old/New，救回/伤害计数。
- `summary.json`：两个读出的 Average/Last；十任务都能配对才给完整互补上界 Average。
- 外层 log 打印 `FrozenReadoutTask` 和 `FrozenReadoutSummary`，发给我即可分析；必要时补传输出目录。

## 判定

优先看 Ridge 是否比 NCM 更好，以及**能救回 DualMask 错例的数量是否增加**。
独立准确率提高但互补救回不增加，不足以支持新专家路由。
互补并集用真实标签选择正确专家，仍是 Oracle 报告，不是合法方法成绩；本轮不新增选择器、不测试拟合路由。
即使互补空间扩大，也需下一步用独立训练侧信号验证“救回多、误切换少”，不能直接宣布突破。
