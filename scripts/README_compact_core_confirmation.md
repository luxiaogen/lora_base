# DualMask最小闭环：只补三组完整T10

固定ImageNet-R T10、seed1993、anchor2.5、20epoch、CA5、math-SDPA，不保存权重。
复用10月6日已成功结果，先核对原始日志、配置、训练源码、软件、物理GPU和数据路径。
旧训练源码参照：`f43ccf986a7a1c18316f46132e64f11841ae7123`。
本轮只增加队列、汇总和测试；跨提交复用表述为“训练源码一致”，不是同提交。

| 机器 | 新训练顺序 | 旧结果复用 | 回答的问题 |
|---|---|---|---|
| 3090 | C10、C00 | R0→C11、R3→C01 | 整套保护权限×幅度抑制的2×2 |
| 5090 | O | S2→M | M相对原完整DualMask的组合精简代价 |

C10/C00仅从R0/R3关闭S/P冲突门。权限关闭仍是双软模式＋保护强度0；
所有3090组均双分支rank64/40、掩码正则0。
O将三个固定控制设为null、P rank0恢复自适应、乘积排名/原数量规则、正则0.01。
O不是已关闭正则并固定控制的F。10%分母为每层完整QKV坐标数。

## 执行

在原conda环境中使用Bash脚本；两台均沿用原实验的`CUDA_VISIBLE_DEVICES=0`：

```bash
bash scripts/10_06_imgr10_compact_core_confirmation_3090.sh
bash scripts/10_06_imgr10_compact_core_confirmation_5090.sh
```

先让3090已有R7完整结束，再拉取新代码。保存本机JSON，不reset或新建worktree。
每台先完成全部新增配置Task0–1的一轮短测，全部通过后进入固定T10队列。
3090约2～2.5小时、5090约50～75分钟；三组完成即结束，不追加候选。
精确参照目录写在`scripts/sweeps/imgr10_compact_core_confirmation.json`，缺失或不匹配即停。
不同GPU的绝对成绩不参与归因。已有参照不复制成新的训练或独立重复。

入口保留`--machine`、`--mode run|smoke|t10|dry-run`、`--modes`、`--resume`。
dry-run只展示新训练命令，不读GPU、写结果目录或启动训练。
续跑示例（必须用实际生成的目录；完整成功组与已验证短测自动跳过）：

```bash
python scripts/run_compact_core_confirmation.py --machine 3090 --resume /absolute/confirmation/queue --modes C10 C00
```

训练异常暂停本机；保留失败证据，修复后新目录只跑未完成项。
单纯汇总/绘图异常保存`analysis_errors.jsonl`，不阻断后续预排训练。
不使用历史PID；`active.json`和`ps`核对当前队列/子进程。

## 结果与解释

输出`logs/shell_logs/imgr10_compact_core_confirmation_<machine>/<timestamp>/`。
含原始训练日志、每组run.json、复用来源及日志SHA、reference_validation.json、
完整/逐任务结果、实际范数/覆盖率/活动参数/成本CSV、Old–New和2×2效应PNG/PDF。
可以离线重算：

```bash
python scripts/analyze_compact_core_confirmation.py /absolute/confirmation/queue
```

匹配失败禁用归因；缺一个完整单元不计算完整2×2。单seed效应仅作描述，
联合成绩更好不自动证明协同；M接近O不证明等效或每个删减项独立无用。
保护权限有益不证明W_pre精确定位旧知识；CA仍保存类别协方差，能力评估成本仍保留。
不自动替换正式基线或启动跨seed/新数据集实验。
