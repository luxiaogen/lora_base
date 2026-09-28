# CA 跨任务难负类间隔：3090 单候选

问题：当前错误不只发生在最后一个新任务与旧任务之间，历史任务之间也有混淆。
测试一个对所有任务类别一视同仁的 CA 目标，而不是单向抬高新类分数。
这是探索性候选，尚无 GPU 性能证据；不是 Oracle、Router 或已经验证的新方法。

## 唯一算法变化

CA 保留原全局 CE，再加：

```text
negative = 当前伪特征在其他任务类别中的最高余弦分数
hinge = mean(max(0, 0.05 + negative - 正确类别余弦分数))
loss = 原 CA loss + 1 × hinge
```

间隔作用于原始余弦分数，不是 CA 温度归一化后的分数。0.05 / 权重1是运行前
固定的探索设置，不是由测试集筛出的最优参数。本次不扫参数。
每类仍采样256条，旧类和新类都作为正类参加，故对称是指监督规则和样本预算相同，
不是保证每对类别恰好同时成为彼此的 hard negative。最高分同任务错误类不会被选为
此项负类，但仍参与原 CE。任务归属由累计 task_sizes 构造，支持不等任务大小；
任务信息仅用于训练伪特征标签，正式推理不需要真实任务 ID。

Task0 不执行 CA，因此不加这项。Task1–9 只更新既有分类头，不回传骨干或 LoRA。
所有伪特征均来自已有训练侧统计；不使用测试混淆矩阵找负类。
双门、CA采样、优化器、训练轮数、merge、推理路径不变。
`ca_cross_task_margin_weight` 默认0，关闭时没有新增随机抽样或候选计算。

## 运行

固定 ImageNet-R T10、seed1993、anchor2.5、20轮、CA5、math-SDPA；
数据路径/device 读取本机 JSON。仅1次完整候选，不重跑基线。
参照本机正在重跑的 `9_28_imgr10_anchor2p5_save_t10_3090`；不能与5090绝对分数相减。

可选独立 GPU 短测（Task0–1各1轮，只验证通路，非性能）：

```sh
bash scripts/9_28_imgr10_ca_cross_margin_3090.sh --smoke
```

正式运行：

```sh
lrun scripts/9_28_imgr10_ca_cross_margin_3090.sh ./logs/9_28_imgr10_ca_cross_margin_3090.log
```

脚本不自动测试、断言或串行短测；`--dry-run`仅显示命令。预计约1.5小时，
基于本机已有81.65分钟T10，新增损失和写盘开销尚未实测。
本次保持 `save_task_weights=true`，每任务保存完整评估权重，见
[权重说明](README_task_weights.md)，建议预留至少6GB。

## 看什么

`CACrossTaskMargin`每轮记录CE、hinge、加权项和active_fraction；只证明损失生效。
性能比较同机完整T10的Average/Last、阶段平均Old/New、Forgetting，以及
StageAudit的旧→旧/旧→新/新→旧计数和CNN with-task曲线。
旧→旧计数包含同任务和跨历史任务错误，不能单独称为跨任务错误。
若只把错误转移到别组，或Average/Last无净收益，不保留；不凭当批CA loss下降判成功。
本候选沿自己的分类头轨迹完成全部任务，不把局部分叉结果拼成正式T10。

验证：独立单元测试检查负类范围、双向梯度、默认关闭、随机流不变、实际CA只更新头、
重跑配方和CLI。CPU测试不等于完整CUDA运行或准确率验证。未连接服务器。
