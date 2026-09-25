# ImageNet-R T10：P 分支学习率配对实验

分支：`codex/mask-budget-comparison-20260924`。只在现有项目修改，不创建新项目。

问题：保留 S/P 冲突门，适度增大 P 的训练步长，能否提高 New 而不损害 Old？
这是待验证的优化假设；不把“梯度受到门控”当成已经确认的性能瓶颈。

## 两次完整训练

| 顺序 | Seed | Task0 | Task1–9 S/分类头初始LR | Task1–9 P初始LR |
| --- | --- | --- | --- | --- |
| 1：baseline | 1993 | 原配方，LR 0.02 | 0.02 | 0.02 |
| 2：candidate | 1993 | 原配方，LR 0.02 | 0.02 | 0.03 |

唯一实验变量为 `plora_lr_multiplier=1` / `1.5`。P 使用同一个余弦调度，保持1.5倍比例。
默认1.0与Task0保留旧的两个优化器组；候选Task1起分成 LoRA(S)/分类头/P 三组。
P未启用时不创建额外组，不改变冻结参数；支持 DataParallel 的 `module.` 名称前缀。
日志 `LoRA optimizer groups` 记录组参数量，`LoRA learning rates` 记录每轮实际LR。

完整继承9/24的3090 A组：AugReg ViT-B/16、QKV、fresh Kaiming、Task0及后续20轮、CA5、
math-SDPA、S/P冲突门、S保护、P可塑区、Energy50/floor10、旧重叠自适应、冲突正则开启、
正则权重0.01。连只读ΔW诊断也保持开启，便于严格配对。没有改JSON默认值或数据路径。
不改变前向掩码、合并、分类头学习率或CA优化器；可训练参数量不增加。

## 执行

先只打印两个正式命令（不训练、不需要数据集）：

```bash
bash scripts/9_25_imgr10_plora_lr_3090.sh --dry-run
```

服务器短测：两组各完成Task0–1，每任务1轮；不是正式成绩，也不能替代完整训练。

```bash
bash scripts/9_25_imgr10_plora_lr_3090.sh --set max_tasks=2 --set init_epoch=1 --set epochs=1 --set ca_epochs=1 --set wandb_mode=disabled
```

正式运行（不要附加短测参数）：

```bash
lrun scripts/9_25_imgr10_plora_lr_3090.sh ./logs/9_25_imgr10_plora_lr_3090.log
```

两组顺序执行，读取本机 `exps/dlora/imgr10.json` 的 `data_path`，不覆盖数据路径。
每组日志与训练prefix带时间戳；个别运行失败会继续下一组，脚本最终返回非零状态。
根据9/24每组约76分钟估计，3090共约2.5–3小时；没有本轮CUDA实测。
测试默认等价性不等于服务器已复现，因此本次保留同机完整基线，不直接将历史分数作分母。

## 结果判定

先确认Task0一致、每个可训练参数只进入一个组、P实际LR为1.5倍，其余组LR一致。
报告完整Average、Last、Forgetting，以及Task1–9每阶段Old/New及其均值。
不把仅New提高、Old降低且总准确率无收益当成改进；只跑seed1993，不继续扫倍率。
当前基线参考为9/24 A组87.008/82.48，但本轮以新配对baseline实际结果为准。

本地检查：

```bash
python -m unittest test.test_plora_lr test.test_night_mechanism test.test_night_mechanism_scripts test.test_global_conflict_budget
```

这些检查覆盖CPU优化器等价性和配置/脚本一致性，不证明CUDA完整训练或性能提升。

本次本地执行：上述37项通过，Python编译及Shell语法检查通过。
全量发现测试128项通过，另有 `test_ca_diagnostics` 因本地缺少 `easydict` 导入失败；
未安装或修改本地训练依赖。真实ImageNet/CUDA短测留给服务器执行。
