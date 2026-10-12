# 最终F单seed确认：实施与接续记录

训练提交已推送至研究分支codex/mask-budget-comparison-20260924：
`d9cb728476259881c7dd9a8039f01cb03953496f`。

用户将三seed验证改为只跑seed1993。共9组，本机配对，不追加候选或改正式基线。

| 机器 | 顺序 | 实验因素 |
|---|---|---|
| 3090 | C11→C01→C10→C00 | 权限×幅度门2×2；C11为当前F |
| 3090 | Soft→Bfixed | 两支均软保护；β固定0.5 |
| 5090 | Lspectrum→Labs→Lperm | 谱、abs(W_pre)、固定行列置换；三位置共同最小有效范数 |

F为ImageNet-R T10，W_pre谱重要性质量覆盖0.90、rank64/64、γ0.5/0.75、非对称权限、α0.5、完整QKV每分支精确Top10%幅度门、β=min[0.5(1+R_old),1]、掩码正则0、20epochs、CA5、math-SDPA、Task0 unmasked/anchor2.5，无checkpoint。原型路由/位置、Fisher、固定P支持、预算α和深度权限均关闭。Bfixed仍执行NCM流程，不能声称已删除其运行开销。

5090三位置逐Q/K/V匹配原谱保护坐标数；同状态按每层、每分支完整QKV缩小到三者最小范数，只缩小、系数不反传、零B时系数1。跨训练轨迹仍须报告范数差异。跨运行保护数量若出现±1偏差，汇总拒绝严格位置配对，不自动重跑。

## 已完成的实施验证

本地16项专项通过；全量935项通过，跳过15项。默认F与d5cb5bc基准的前向、梯度、一次优化步和合并逐项一致；Task0、零B梯度、P硬权限、三位置数量/RNG/范数、前向合并一次和队列截止/失败/分析隔离均有测试。独立代码复审通过。

审查发现并修复两类只读问题：软权限P合并诊断不能继续假设硬补集；汇总必须核对实际Top10%数量、合并证据和机制字段。训练/实际合并公式未因诊断修复而改变。norm控制的额外移除量不冒充纯冲突移除量。

5090专项16项通过。全量926项检查遇到两项既有本机差异：test_baseline_suite要求本地JSON的CA5而旧JSON为10；可选test_feature_geometry缺sklearn。保留JSON和环境不改，排除上述两项后适用924项通过，15跳过。本轮命令显式CA5；这不是宣称5090原全量无例外通过。临时适用入口首次因导入目录不正确而失败，修正路径后通过，未启动或重试任何训练。

## 2026-10-13 01:06启动前快照

3090原项目`/home/shengqin/lys/baseline/LoDA_ICML2026`，SSH shengqin3090ti，RTX3090 Ti。另一已授权深度权限四组整队正在运行，外层Python队列PID1903745，源码fa53a839d4a8df7edd28afdf230b577b548179b6，目录`logs/shell_logs/depth_permissions_3090/20261013_004924_345406`。本轮未更新其正在使用的源码。

- 本轮Fish lwait PID：1920516；不是新训练队列PID。
- 接续总日志：`logs/10_13_final_f_validation_3090_lwait_d9cb728.log`。
- 临时入口：`/tmp/dualmask-final-f.AFS4bT/wait_3090.fish`。
- 执行的等待：`lwait 1903745 "bash /tmp/dualmask-final-f.AFS4bT/after_queue_3090.sh"`。
- 旧整队退出后，先核对四组完整T10、200epochs、有限指标、源码指纹和GPU空闲，再精确快进到训练提交、运行服务器CPU回归，最后启动本轮六组。任一步失败停止，不自动重试。

远端Git只读预检发现3090公钥认证不可用。未更改凭据，改为上传已推送提交的Git bundle至`/tmp/dualmask-final-f.AFS4bT/final_f.bundle`；bundle前置提交为当前fa53a83，包含d9cb728。回调在旧队列成功结束后才fetch本地bundle并精确merge到d9cb728，不把bundle中附带的文档提交当作训练版本。

5090原项目`/mnt/disk1/lys/CIL/code/baseline/lora_base`，SSH 5090d，RTX5090D，Python3.10.20/torch2.7.1+cu128/timm0.6.12。当前项目没有训练进程，已精确快进到上述训练提交；五份本机数据JSON的SHA256前后逐项一致。GPU上另一个独立项目进程PID3634279仍在运行，没有停止或修改该项目。

- 本轮Fish lwait PID：3655882；不是新训练队列PID。
- 等待总日志：`logs/10_13_final_f_validation_5090_lwait_d9cb728.log`。
- 临时入口：`/tmp/dualmask-final-f.Q3oRIl/wait_5090.fish`。
- 执行的等待：`lwait 3634279 "bash /tmp/dualmask-final-f.Q3oRIl/start_5090.sh"`。
- 退出后再次检查项目SHA、未提交代码和GPU计算进程；如有其他任务接着占用GPU则暂停，不并发挤占。空闲才进入三组短测及正式队列。

两机均无定时任务。各自从首个GPU短测开始计算10小时启动上限，已经启动的T10完整结束。每台全部预排配置先Task0–1各一轮GPU短测；全部通过才正式训练。短测或训练失败暂停，分析错误单独保存。

本快照本轮完成数为0/9，尚无GPU短测结果、正式训练PID或正式运行目录。这些需在等待结束后从manifest/active/queue及ps回填，不能复用上述旧PID或根据lwait退出断言成功。

入口为`scripts/10_13_final_f_validation_3090.sh`与`scripts/10_13_final_f_validation_5090.sh`。预计新队列3090约7小时、5090约3–5小时，不包括等待；首个完整组后由日志校正。

[实施计划](../plans/2026-10-13-final-f-validation.md)；[实验卡](实验日志/DualMask/最终F确认/DM-B-261013-002.md)；[启动证据](raw/experiments/2026.10.13_最终F确认_DM-B-261013-002/来源.md)。
