# 固定保护强度：3090单组精简验证

固定保护强度alpha=0.5，保留冲突强度beta=min(0.5*(1+R_old),1)，
R_old=max(0,C_new-C_all)。覆盖重要性质量0.90、S/P rank64/64。
ImageNet-R T10、seed1993、20epochs、CA5、math-SDPA；Task0 unmasked/anchor2.5。
原谱保护位置、非对称权限、gamma0.5/0.75、全层QKV幅度精确Top10%、step、
掩码正则0、不保存checkpoint，保留相同只读原型位置诊断。

复用固定保护强度及direct_strengths开关，Task1起跳过D_t诊断；
scaled冲突模式保留原R_old公式。Task0仍计算原D_t、使用原初始化路径。
模型、分类目标、训练前向和最终合并公式均沿用已有实现。

## 配对参照

DEMAND：`ae27470cb136db679be545ae3fd7f691574cabdc`，
`logs/shell_logs/imgr10_ncm_demand_restore_3090/20261010_161939_092651/DEMAND_seed1993`。
完整健康T10：Average87.247、Last82.55、Old82.11、New86.70、Forgetting5.5611；
Task097.10、StageOld86.4633、StageNew86.3467。
本组改为固定alpha并删除增量阶段D_t计算；其余有效训练配方一致。
配置差异为fixed_protect_strength、ncm_direct_strengths及wandb_group/prefix。

OVERLAP：`6e3abdd4336d1aaee1e2df86a0ce3e2cb9f37f55`，
`logs/shell_logs/imgr10_ncm_overlap_strengths_3090/20261010_144150_947509/OVERLAP_seed1993`。
它同样跳过增量D_t，但alpha=C_new；本组相对它只改变fixed_protect_strength，
另有wandb_group/prefix差异。Average87.167、Last82.47、Old82.18、New85.15。

实施及运行后核对训练源码哈希、软件、硬件和配置；披露跨提交、训练源码一致。
M90还固定beta=0.5，不能把M90成绩当作本次固定alpha的单变量证据。
单seed只用于精简方向判断，不宣布等效或自动更换基线。

## 执行

入口：`bash scripts/10_10_imgr10_ncm_fixed_protection_3090.sh`。
仅FIXED_seed1993，先Task0–1各一轮GPU短测，成功后完整T10。
短测或训练失败暂停、不自动重试；分析错误独立记录。
使用全新目录，3小时仅为新组启动上限；预计正式训练约66分钟。
仅使用3090原目录并保留服务器数据JSON；空闲直接启动，繁忙时核实外层PID后lwait。

本地4项新增专项测试及12项相邻控制回归通过；全量轻量测试822项，跳过15项，其余通过。
独立审查44项相关测试通过、无阻断；编译、Bash语法、单组配置及dry-run通过。
服务器启动前复核实际FixedTaskControls；任务开始的competence日志是未覆盖的控制器建议值。
