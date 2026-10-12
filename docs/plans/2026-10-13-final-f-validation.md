# Final F single-seed verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 按用户最新选择，只用seed1993完成最终F的核心2×2、权限、β与三位置受控对照。

**Architecture:** 复用现有短测先行、指纹、截止和续跑队列；仅新增默认关闭的三位置整层共同最小范数控制。所有归因限本机；历史M/O不替代本轮F。

**Tech Stack:** Python、PyTorch、JSON sweep、SSH、Fish lwait。

---

## 1. 已授权范围

用户将三seed改为单seed1993，共9组，不追加参数搜索。两台各最多10小时启动窗口，从首个GPU短测开始计时，已启动完整T10跑完。预计3090约7小时、5090约3–5小时，实际由首个完整运行校正。

F：ImageNet-R T10，20轮，CA5，math-SDPA，Task0 unmasked/anchor2.5，不保存checkpoint。W_pre谱重要性质量覆盖0.90（不是坐标比例），S/P rank64/64，γ0.5/0.75，S软保护/P硬补集，α=0.5，完整packed QKV每支精确Top10%幅度门，β=min[0.5(1+R_old),1]，掩码正则0。原型分配、原型位置、Fisher、固定P支持、预算和深度调节均关闭。

固定β组仍执行NCM只读评估；不能据此声称已删除其运行成本。掩码正则0不关闭Task0独立anchor。

## 2. 固定队列

3090原项目/home/shengqin/lys/baseline/LoDA_ICML2026，SSH shengqin3090ti：

| 顺序 | 组 | 仅改变因素 |
|---|---|---|
| 1 | C11 | F同提交参照 |
| 2 | C01 | symmetric_soft＋α=0，两支全坐标权限 |
| 3 | C10 | S/P冲突门关闭 |
| 4 | C00 | 合并C01/C10因素 |
| 5 | Soft | symmetric_soft，α仍0.5 |
| 6 | Bfixed | β固定0.5 |

5090原项目/mnt/disk1/lys/CIL/code/baseline/lora_base，SSH 5090d：

| 顺序 | 组 | 位置 |
|---|---|---|
| 1 | Lspectrum | W_pre谱 |
| 2 | Labs | 每Q/K/V同数量最大abs(W_pre) |
| 3 | Lperm | 谱掩码固定行列置换 |

三位置均启用prior_layer_min：同一原始BA下，三套权限乘相同幅度门，按每层、每分支完整QKV取三者范数最小值，较大者缩小，系数detach，零增量系数1，最后乘原γ。不是跨轨迹绝对等量，实际轨迹仍须报告。置换独立RNG且跨任务固定。

保护数量在各运行内部逐Q/K/V严格匹配；跨运行若再现±1偏差，在汇总保留问题且不输出严格位置配对，不自动重跑。

## 3. 实施与验证

新增scripts/run_final_f_validation.py、analyze_final_f_validation.py、sweeps/final_f_validation.json、两机Bash入口及test/test_final_f_validation.py。

必要模型修改仅为三位置缓存与有效门统一缩放；旧paired_min、paired_layer_min及默认路径保持原行为。日志将范数匹配移除量与冲突单独移除量分清，前向/合并使用同一门。

1. 先写失败测试，再实现：9组差分、Task0、默认F前向/损失/梯度/一步优化/合并、零B梯度、P硬权限、三位置数量/RNG/范数、合并一次。
2. 测汇总缺失单元、指纹/配置不匹配、训练失败暂停、分析失败继续、截止/续跑；设置检查仅入口和test。
3. 本地完整轻量测试与只读代码审查后，仅提交本轮文件/改动，不纳入其他未提交的深度实验。
4. 推送实际SHA，两机安全快进，保留本地五份JSON及数据路径；不新建worktree，不改正在运行的代码。
5. 启动前核对GPU和最新外层队列。3090有活动队列时用已核实PID的Fish lwait；旧队列完整成功后才接续。若无活动队列，检查完成记录后直接启动，不等待历史PID。不设置定时任务。
6. 各机全部预排配置先Task0–1各一轮GPU短测，全部通过才正式T10。失败暂停，不自动重试；纯分析错误不阻断训练。

当前00:15前后只读快照未发现3090训练或计算进程；5090只有桌面进程，五份数据JSON修改保留。部署时必须再次核对，不能把该快照当实时保证。

## 4. 输出与解释

保存原始日志、run.json源码/配置/软件/GPU指纹、queue/active/manifest与期限、逐任务/epoch/掩码/实际更新/成本表。报告Average、Last、最终和阶段Old/New、Forgetting、参数数和显存。

只有完整匹配单元进入2×2效应或配对；不以单seed报告标准差、显著或等效。C11联合最好不自动称协同。Soft接近则非对称退居设计选择；Bfixed接近则β自适应列为删减候选。谱/abs接近不证明对所有评分不敏感，也不能根据位置结果单独认定P写入机制。只读换掩码有训练适应偏向，尤其P硬权限，须披露。

本轮不替换正式基线，不自动新增seed或数据集。全部9组完成后再决定最终最小方法与论文可保留主张。
