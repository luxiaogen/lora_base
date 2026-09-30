# P 可塑区硬置零：随机／幅度／冲突三组

状态：已实现，本地单元与脚本验证通过；未启动 GPU 实验，不预设性能有效。

## 问题与变化

此前 P 冲突门渐进加强的完整 T10 为 Average 87.151、Last 82.38，
相对最新同机 layer 基线分别 −0.059pp、−0.07pp；仍未改善 Old–New 取舍。
本轮执行此前第二项未做方案，检验禁止部分 P 更新是否能促使模型利用剩余位置学习。

**只替换 Task1–9 的 P 冲突门**：先应用原 P 可塑区限制，再将其中 40% 坐标硬置零。
不叠加原 P 软冲突门；S 分支继续用原保护门和软冲突门。前向、评估、merge 都使用同一硬门。

```text
P 有效更新 = 原 P 更新 × P 可塑区 × (1 − 硬置零掩码)
```

各层在合并的 QKV 矩阵中共同选点，数量为 `floor(0.4 × 本层 P 可塑坐标数)`；
本来就被保护区排除的位置不计入分母。不是整个 QKV 的 40%，也不是最终 ViT 权重稀疏率。
实际零更新比例可能更高：未选坐标本身也可能为零。LoRA 可训练参数量及稠密计算量不因此降低。

| 顺序 | 组别 | 在 P 可塑区中删除哪些坐标 |
| --- | --- | --- |
| 1 | random_p40 | 均匀随机选定相同数量；每层、每任务固定，不按 batch 重抽 |
| 2 | magnitude_p40 | 按当前 `abs(BA)` 从大到小选 |
| 3 | conflict_p40 | 按 `W_pre importance × abs(BA)` 从大到小选 |

随机组使用独立 Generator，由 seed1993、任务和层编号确定，不消耗训练全局随机数。
随机布尔掩码仅为当前任务的非持久缓冲，merge 后清空；不保存每任务稠密增量或权重文件。
幅度／冲突组的选点随当前 BA 改变，与原动态门的习惯一致。
同一步正的分支 gamma 缩放不改变排序；前向与 merge 选点一致由测试验证。

## 固定协议与参照

3090 原项目，AugReg ViT-B/16，ImageNet-R T10，seed1993。
anchor2.5，Task0／增量各20轮，CA5，math-SDPA，原 rank 自适应、学习率、分类目标及正则公式。
Task0 不改变；关闭前一轮 `p_conflict_strength_warmup`，不混用其他候选。
数据及预训练权重路径读取本机 JSON，不写入脚本。

三个候选只在 `p_hard_zero_mode=random/magnitude/conflict` 上不同，
`p_hard_zero_ratio=0.4` 相同。不重跑基线、不保存 checkpoint。
完整配置见 [sweep](../../scripts/sweeps/imgr10_p_hard_zero_3090.json)。

复用 `/Users/luxiaogen/Desktop/loda_logs/9-30/9_29_imgr10_granularity_budget_refresh_3090.log`
中正式 `layer_budget1_seed1993` 段：Task0 97.10、Average87.210、Last82.45、
Old82.05、New86.18、Forgetting5.6867。基线软门与候选硬门的范围和强度都不同，
因此基线比较检验整项候选；三种硬门之间才是同数量的选点对照。

## 执行与判定

```bash
bash scripts/9_30_imgr10_p_hard_zero_3090.sh
```

自动先按顺序做三组短测：Task0／Task1 各1轮、CA1、W&B offline；
全部短测通过才进入三组正式 T10。短测分数不用于性能结论。
日志分开按时间戳、smoke/run、PID 保存，不覆盖历史运行。
正式三组按最近同机约80分钟／组估算，共约4～4.5小时，共享负载另计。

每层 merge 输出 `PHardZero`：规则、P 可塑坐标数、硬置零数量和比例、
保留的非零坐标数、移除更新范数、门前范数及 merge 一致性误差。
`P applied merge diagnostic` 也读取实际硬门，`applied_strength=1`。
不在训练路径新增 assert／输入防御性验证，正确性检查放在 test。

同时看 Average、Last、任务1–9平均 Old/New、Forgetting。
New 单项上升但 Old 下降、整体无净收益不能当作有效提点。
同样删除数量不等于同样移除范数，必须一并报告；若冲突组未优于幅度／随机，如实记录。
单 seed 是初步性能证据，不是显著性结论。

本地8项专项测试、76项相关测试通过；编译、Shell语法、dry-run配置核对通过。
完整发现370项中368项通过，另两个模块因本机缺少 `easydict` 未加载，
不宣称完整回归全通过。独立代码审查未发现指定实验范围内的阻塞问题。
CUDA短测尚未执行；上述验证不等于端到端训练成功或性能有效。

## 与 P-Oracle 的边界

早期 P-Oracle 用真实任务 ID 调节历史 P 分量，只能作诊断上限；
这次硬门不读取测试标签／真实任务 ID，不增加路由或推理前向。
不能将其称作 Oracle 复现，不能将历史 Oracle 收益叠加到当前基线。
预测任务的路由虽可合法使用，但此前已试过，并需额外统计路由错误、存储和推理成本；
本轮不重开该路径。训练期任务标签本身不是作弊，测试期用它选择 P 分量则属于 Oracle。
