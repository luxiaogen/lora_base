# DualMask 精简与结构：3090＋5090 固定夜间队列

本轮为已批准的组件删减与归因实验，不新增损失/教师，不预先替换正式基线。
研究分支为 `codex/mask-budget-comparison-20260924`；纯基线调试分支不变。

## 公共配方与分母

ImageNet-R T10，seed1993，Task0/incremental各20 epochs，anchor2.5，CA5，math-SDPA。
沿用原优化器、学习率和Task0路径；Task1起A冻结、B训练。
无旧训练图片回放、真实任务ID推理或checkpoint保存。
精确矩阵见 `scripts/sweeps/imgr10_compact_structure_night.json`；其余固定设置继承
`scripts/sweeps/imgr10_protect_position_3090.json`，不会覆盖服务器数据路径JSON。

- F（旧B3配方的同提交参照）：重要性质量覆盖目标0.825，保护强度0.5，S/P rank64/40，
  冲突强度0.5，全部掩码正则权重0，Task0独立anchor保留；原能量选点数量和乘积排名。
- M（预先指定的精简候选）：F上改为固定10%幅度排名。精确Top-K分母是**每层完整QKV**，
  S/P分别选点，不是P可塑区10%。取整为floor；并列值由Top-K选足数量，启动零B时仍有梯度。
- 0.825不是保护82.5%的坐标。固定控制组仍执行能力评估，不能宣称删除了它的计算成本。
- S4/S5用原乘积决定数量，再按幅度排名；只有M的选点与数量都不再需要乘积。

## 固定顺序

| 3090 | 配置 | 比较 |
|---|---|---|
| R0 | M双分支非对称 | 新的精简参照，不重跑原始大基线 |
| R1 | M两支均软保护 | R1−R0：P硬权限是否必要 |
| R2 | M单分支rank104软保护 | R2−R1：分支组织＋合成/分别门控 |
| R3 | M双分支无保护权限 | R3−R1：保护权限作用，冲突门不变 |
| R4 | M原位置paired_min | R4−R5：同状态范数受控的位置配对 |
| R5 | M置换位置paired_min | 固定独立行列置换，训练RNG不变 |
| R6 | M置换位置不控范数 | R0−R6；R5−R6：完整策略与范数控制 |
| R7备用 | M单分支无保护权限 | R7−R3：无权限时分支组织的作用 |

R3/R7用两支软权限＋保护强度0实现全坐标可改，不关冲突门。

| 5090 | 配置 | 比较 |
|---|---|---|
| S0 | F | 新补一次B3同提交参照 |
| S1 | F固定10%乘积 | S1−S0：原数量规则是否必要 |
| S2 | M固定10%幅度 | S2−S1：相同数量下乘积作用 |
| S3 | F仅恢复冲突强度自适应 | S3−S0 |
| S4 | F原数量、幅度排名 | S4−S0；S2−S4 |
| S5 | S3＋原数量、幅度排名 | S0/S3/S4/S5排名×强度2×2 |
| S6 | M均匀移除范数匹配 | S2−S6：选择位置与统一缩小 |
| S7备用 | M自适应冲突强度 | S7−S2 |

两机分别归因，不作跨GPU绝对分数比较，不因准确率低跳过预排组。

## 单分支的实际含义

默认新增选项 `dual_mask_branch_layout=dual`；只有Task1+的`single`改变布局。
先按原顺序构造S64/P40，复用全部初始化抽样，然后拼接：

```text
A_single = concat_rows(gamma_S * A_S, gamma_P * A_P)
B_single = concat_columns(B_S, B_P)
```

B零初始化、A冻结，拼接后总rank104、外部gamma1；临时P释放，不额外构造随机模块。
保持无门控时原始更新和B梯度尺度，但合成更新只应用一套权限/冲突门。
因此本轮因素是“分支组织与门控方式”，不能把结果全部归因于分支数量。
训练/最终合并一致，临时更新只合并一次。

## 执行与续跑（服务器Fish中使用Bash脚本）

安全拉取代码时保留本机未提交的JSON；不要reset、覆盖路径或新建worktree。
在实际训练环境激活后：

```bash
bash scripts/10_06_imgr10_compact_structure_3090.sh --hours 9
bash scripts/10_06_imgr10_compact_structure_5090.sh --hours 9
```

`--mode dry-run`仅展开命令，不调用GPU或写运行目录。
默认run先执行本机全部8组Task0–1一轮短测；全部通过后进入完整T10。
从首个GPU短测开始计9小时，截止不启动新组，已经开始的组完整结束。
预算未启动组标记`time_budget_pending`，不是失败/负结果。
预计3090约6–7组、5090约5–6组，实际以首个完整组耗时校正，不保证全部完成。
训练异常暂停该机；纯汇总/画图异常写`analysis_errors.jsonl`后继续固定训练顺序。

输出目录为 `logs/shell_logs/imgr10_compact_structure_<machine>/<timestamp>/`：
`manifest.json`记录队列PID和提交；`active.json`记录当前训练PID、阶段和日志。
每组有`run.json`（有效配置/源码SHA256/机器软件环境）和`training.log`。
不要把历史PID用于lwait；以本机最新active.json及`ps -p PID -o pid,etime,args`核实。

已完成组不重跑，预算耗尽后可显式授予新窗口续跑同一目录：

```bash
bash scripts/10_06_imgr10_compact_structure_3090.sh --resume /absolute/queue/directory --modes R6 R7 --hours 9
```

续跑要求同机器/提交/源码/配置且成功结果可核验；不会覆盖已有运行目录。
续跑前还须核对原conda/Python、torch/torchvision/timm/numpy版本和GPU UUID与run.json一致；
脚本不在续跑前自动查询这组身份，汇总发现环境变化会禁用归因对比。
失败或中途丢失的组保留原证据，修复后用新目录和`--modes`指定未完成组。
`--mode t10`用于已经独立完成短测的人工续接，首次执行应使用默认run。

## 结果与固定解释

运行器每组结束自动生成results.csv/json、report.md、contrasts.json、逐任务/范数/覆盖率/
诊断/成本/持久张量与活动参数量CSV，以及Old–New、位置margin、范数、结构和成本PNG/PDF。
可离线重做汇总：

```bash
python scripts/analyze_compact_structure.py /absolute/queue/directory
```

只对完整、同提交、匹配配置的同机配对计算归因；短测与部分T10不算正式性能。
同状态paired_min不等于两条训练轨迹处处同范数；均匀模式匹配冲突移除范数，二者不同。
诊断标签只报告，不训练/筛参数；CA仍保存类别协方差，储存统计不含优化器/临时workspace。

- 删除组件后接近：精简候选，不以单seed宣称等效。
- New上升却明显伤Old/Last/遗忘：取舍，不自动采用。
- 幅度与乘积接近：支持简化排名，不代表W_pre保护位置无用。
- 单/双分支接近：认真考虑删除分支复杂性，不继续凭名称强调创新。
- 位置优势不一致：收缩保护位置主张，不追加评分搜索。

| 组件 | 本轮证据 | 保留理由/收缩边界 |
|---|---|---|
| 数量/乘积规则 | S0/S1/S2/S4 | 真正不依赖乘积的是M；小差异仅列删减候选 |
| 最后冲突强度控制 | S0/S3/S4/S5、S2/S7 | 匹配完整对照后判断，不增加控制器 |
| 保护权限/位置 | R1/R3、R4/R5、R0/R6 | 披露实际更新量与位置诊断是否一致 |
| 双分支与门控组织 | R1/R2、R3/R7 | 同rank/原更新尺度，但门控组织变化 |
| 选择性冲突抑制 | S2/S6 | 只匹配额外移除范数，不等于所有干预相同 |
| 掩码正则 | F/M统一关闭 | 使用旧实验删减线索，本轮不重复扫系数 |

最终目标是确定值得进一步确认的最小配置；本轮不自动替换正式基线。
