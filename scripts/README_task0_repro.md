# Task0 重复性诊断（3090）

三次独立进程，ImageNet-R 原 T10 类别划分，只训练 Task0；每次 seed1993、20 epochs。
只改变运行名称，不改初始化、DataLoader、损失、合并或确定性开关。
使用本机 `exps/dlora/imgr10.json` 的数据路径和其余配置；显式固定 CA5、reg weight=0.01、layer。
Task0 不执行 CA。当前 JSON 的 conflict regularizer 开关保持不变。

先运行一轮 1 epoch smoke（不用于三次正式对比）：

```bash
python main.py --config exps/dlora/imgr10.json --set 'seed=[1993]' --set max_tasks=1 --set init_epoch=1 --set task0_repro_diagnostic=true --set prefix=task0_repro_smoke
```

检查出现 `Task0Repro` 的 initial、first_batch、epoch、final 后，执行正式三次重复：

```bash
bash scripts/9_24_imgr10_task0_repro_3090.sh
```

不要同时启动其他训练。预计 3090 三次共约 20–25 分钟（参考 9/24 Task0 训练375秒＋评估约6秒，不含异常I/O等待）。
独立日志在 `logs/shell_logs/imgr10_task0_repro_3090/`。

按以下顺序对比三次 `Task0Repro` JSON 记录：

1. environment / dataset / class_order：源码、运行环境、样本路径与标签顺序是否一致。
   manifest 不读取图片字节，不能替代数据内容一致性审计。
2. initial：完整模型（含 buffer）、LoRA、分类头和 CPU/CUDA RNG 哈希。
3. 每轮 first_batch：样本编号、标签、随机增强输入的哈希；不新增数据遍历。
4. 每轮 epoch：loss、训练准确率、可训练参数哈希。
5. final：原评估流程返回的 Task0 准确率，以及合并后完整模型哈希。

首批相同不能证明整轮所有批次相同；若后续权重才分叉，需进一步定位数据或计算。
本轮没有每轮测试集评估、没有保存可续训 checkpoint，也不挑测试准确率最高的重复。
观察功能默认关闭。单元测试验证哈希过程不改变 CPU RNG、参数和简化训练结果；CUDA完整复现需要这三次服务器实验验证。

## 首次分叉的批次定位

上面三次完整训练的初始权重及每轮首批输入一致，但第 1 轮的最终权重不同。
下一步仅跑 1 个 epoch、重复两次，按批次输出 `Task0Repro` 的 `batch` 记录：
样本序号哈希、增强后输入哈希、该批 loss 和完成 optimizer.step 后的可训练参数哈希。

```bash
bash scripts/9_24_imgr10_task0_batch_repro_3090.sh
```

日志在 `logs/shell_logs/imgr10_task0_batch_repro_3090/`。按 batch 序号比较两个日志：
若输入先不同，检查数据加载与增强；若输入相同但更新后的参数不同，
下一步检查该批前向、反向及 GPU 算子。哈希会让 GPU 参数同步到 CPU；
这轮只用于定位，不作为性能实验。
