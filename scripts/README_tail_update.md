# 连续尾部抑制实验

## 算法与对照

每层、每分支在未乘γ的packed QKV BA中选择精确Top-10%；τ为其中最小绝对值。C用
`d - 0.5 * sign(d) * relu(abs(d) - τ)`替代阶梯抑制，再应用权限和γ。
τ不求导；绝对值超过τ的梯度为0.5，等于或小于τ为1。Task0沿用原路径。

T保留原Top-K位置，U统一缩小权限允许的更新，两者按同状态C的权限后移除范数计算系数。
系数不求导。只匹配每层每分支同一状态的范数，不保证不同训练轨迹始终等量。
`dual_mask_update_rule`默认`step`；新增模式用于本计划固定的layer/global-P、原非对称权限、无位置范数控制配方。

M固定质量覆盖0.825、保护强度0.5、抑制强度0.5、幅度Top-10%、掩码正则0。
10%分母为完整层QKV坐标，0.825指重要性质量。ImageNet-R rank64/40，ImageNet-A rank32/20。
数据集各自的学习率、margin、γ和数据路径由本机原JSON提供。anchor2.5、CA5、math-SDPA；不保存权重。

## 运行

3090默认9组，顺序为每个seed的M→C→T，seed依次1993/1996/1997。
5090脚本已准备：ImageNet-A每seed O→M→C→T→U，之后每seed M40→C40，共21组。
**用户最新指示：5090暂不启动。**

```fish
bash scripts/10_07_tail_update_3090.sh
# 配置预览，不启动GPU、不生成文件
bash scripts/10_07_tail_update_3090.sh --mode dry-run
# 用原队列续跑（代码、配置及已完成证据须一致）；保留原10小时截止
bash scripts/10_07_tail_update_3090.sh --resume /实际队列绝对目录
```

队列先对每种配方做一次Task0–1、各一轮短测；不重复短测三个seed。
10小时从本次首个GPU短测开始计时，截止后不启动新正式组，已启动T10完整结束。
短测或训练失败暂停本机；分析错误写入`analysis_errors.jsonl`，后续训练继续。
已成功且通过指纹验证的组不重跑。另一个进程仍在使用该队列时不得续跑。

40轮配方只改Task1+的epochs，Task0为20轮，余弦调度周期自动随epochs改变。
三seed不完整时，只报已有逐seed配对，不报告完整均值。

## 可追溯结果

输出为`logs/shell_logs/tail_update_3090/时间戳/`：

- `manifest.json`、`budget.json`、`queue.json`、`smoke_queue.json`、`active.json`：顺序、原截止、状态、PID、当前日志。
- 每组`run.json`、`training.log`：有效配置、源码哈希、软件/GPU身份、日志。
- `results/aggregate/pairs.json`及CSV、`tasks.csv`、`tail_updates.csv`。
- `paired_accuracy`、`old_new`、`intervention_and_gate_change`三张PNG/PDF图。
- `historical_original.json`：旧O指纹核对；源码/诊断不同只列历史参照，不自动补跑O。

只读采样在Task1–9的epoch1/5/10/20/40、前两步更新后记录。
每层分支Q/K/V记录原始、权限后、有效和移除范数；τ、Top-K覆盖率、实际施加覆盖率分开。
保留最多2048个固定坐标的门系数至下一步，随后释放；没有图片、历史特征或完整门矩阵缓存。
峰值显存由CUDA报告；实际时间来自子进程墙钟时间。

对C−M的预定跟进条件：三seed平均Average≥+0.2pp、Last≥0、至少两seed改善、
最差seed Average≥−0.2pp、阶段Old≥−0.1pp。只是实用门槛，不是显著性检验。
候选有效后仍须看C−T/C−U：成绩和实际抑制量共同支持解释。
