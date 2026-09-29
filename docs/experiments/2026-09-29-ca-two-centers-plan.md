# 两子中心CA：两机各一个完整T10候选

状态：已实现，正式GPU结果待运行。不是已证明有效的方法。
基于提交 `1bceb48`，原项目、原分支 `codex/mask-budget-comparison-20260924`。

## 问题与边界

检验每类单高斯近似是否遗漏有用的类内结构。此前中心迁移、协方差收缩、
真实新类特征CA、当前任务样本区分均未提供可靠整体净提升；不能将这些负结果
写成“多峰已被证明是瓶颈”。本次不加样本区分损失，不扫中心数量，不重跑历史基线。

## 唯一改动

`ca_two_centers=true`（默认false）：每类两个子中心，共用一份类内残差协方差。
数据是正常 `_compute_class_mean` 已提取的当前任务训练特征；从Task0开始统计，
Task0仍不执行CA、不改变训练损失或初始化。旧类只复用当时保存的统计，不重提取旧图片。

固定两中心、最多20次Lloyd更新、确定性最远点初始化，不消耗训练RNG。
退化为同一团时，一个分量可以权重为0；不强行制造不存在的第二团。
不增加训练路径的assert/参数验证框架；算法测试均在test目录。

设n个特征的总体均值为mu；簇中心为mu_k，权重p_k=n_k/n，偏移d_k=mu_k-mu。
令E为各样本相对所属簇中心的残差，B=sum(p_k*d_k*d_k^T)，存储：

```text
shared_cov = (E^T E + B) / (n-1) + 0.001 I
```

有限样本修正B/(n-1)使 `shared_cov + B` 等于原始无偏样本协方差加同样的jitter。
因此改变的是分布形状，不是额外缩小总体方差。公式匹配的是分布期望，
不是承诺每次随机抽取的256条样本拥有逐位相同的均值/协方差。

CA保留原有类龄均值缩放a，采样为 `a*mu + d_k + noise(shared_cov)`。
只缩放整类mu，不缩放d_k，否则会意外改变类间分量协方差。
类别内按p_k随机选分量，每类总计256条，而非每分量256条。
分量选择用独立CPU generator；高斯抽样与shuffle维持原全局RNG调用数量。

开启时 `_class_covs` 保存shared_cov，不另存一套全协方差；新增
`_ca_mixture_offsets`、`_ca_mixture_probs`，200类768维float32共约1.17MiB。
旧统计陈旧性仍存在，本方法不声称修复表征漂移。
此次脚本禁用统计迁移、真实新特征、协方差收缩、额外margin、head-balance等候选，
未验证与这些候选叠加；不默认承诺组合兼容。推理不变、仅CA分类头训练变化。

## 分配与对照

| 机器 | anchor | 正式任务 | 额外候选 | 同机历史参照 |
| --- | ---: | --- | --- | --- |
| 3090 | 2.5 | ImageNet-R T10 seed1993 | 两子中心CA | anchor2p5_save：Avg87.281 / Last82.67 |
| 5090 | 5 | ImageNet-R T10 seed1993 | 两子中心CA | old_competition的weight0基线：Avg87.085 / Last82.67 |

3090参照原始日志：`/Users/luxiaogen/Desktop/loda_logs/9-28/9_28_imgr10_anchor2p5_save_t10_3090.log`。
5090参照原始日志：`/Users/luxiaogen/Desktop/loda_logs/9-26/9_26_imgr10_old_competition_5090.log`。
均20epochs、CA5、math-SDPA、layer双门、冻结增量A、自适应P-rank，机器JSON负责data_path/device。
两台不是多seed；历史对照不是同提交重跑，微小差异不能包装成可靠提点。

## 执行

3090原目录：`/home/shengqin/lys/baseline/LoDA_ICML2026`。
5090原目录：`/mnt/disk1/lys/CIL/code/baseline/lora_base`。

```bash
# 3090
lrun scripts/9_29_imgr10_ca_two_centers_3090.sh ./logs/9_29_imgr10_ca_two_centers_3090.log
# 5090
lrun scripts/9_29_imgr10_ca_two_centers_5090.sh ./logs/9_29_imgr10_ca_two_centers_5090.log
```

每个脚本默认先一组Task0–1各1epoch执行短测、CA1、W&B offline；正常退出后
进入本机唯一完整T10候选。短测不按准确率晋级，也不调用性能相等assert或预检测试套件。
可用 `--smoke` 只短测，`--full` 只正式，`--dry-run` 只打印；拼错模式不会启动训练。
短测和正式均显式 `save_task_weights=false`、`dual_mask_vis_save_weight=false`。
不保存模型权重、不删除现有checkpoint。正式与短测前缀分开，记录实际提交与配置。

3090历史基线约79分钟，预计单候选含短测约1.5–2小时；5090共享负载会变化，
最近完整T10约101分钟，但不能保证相同耗时。此处不是为了凑满8小时的重复网格。

## 判读

- `CATwoCenters`逐类报告样本数、分量比例、between_trace_fraction，确认不是全部退化单中心。
- `CA feature source: new=two_center old=two_center per_class=256`确认旧新两侧均启用。
- 比较Average/Last、逐任务Old/New、Forgetting与错误转移；不能只报告New改善。
- 无旧图片回放；没有测试集参与拟合。现有测试图仅提出假设，不按其中指定类别对做特殊处理。
- 训练随机状态与默认路径数值一致性有CPU测试，不保证跨GPU逐位一致。

## 验证记录

已完成6项算法/真实CA方法CPU测试、3项实际脚本执行测试；独立代码审查通过。
Python编译、两机脚本Bash语法和diff检查通过。
本地完整回归338项：336项通过，2个既有模块因缺少easydict导入失败，
分别为test_ca_diagnostics和test_task0_margin_screen；不称完整回归全绿。
GPU短测由用户启动脚本后执行，当前不声称正式结果或GPU验证已完成。
