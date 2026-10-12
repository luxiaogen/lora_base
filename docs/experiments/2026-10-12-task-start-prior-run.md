# 任务起点权重双机运行记录

实验卡：[DM-B-261012-001](实验日志/DualMask/任务起点权重/DM-B-261012-001.md)。训练提交56cb2fda011ea9d13eb954abdbd74606df0bc540，已推送codex/mask-budget-comparison-20260924。这是启动记录，尚非完整T10性能报告。

公共配置：ImageNet-R T10 seed1993、20轮、CA5、math-SDPA、Task0 unmasked/anchor2.5、无权重保存。W_pre质量覆盖0.90，S/P rank64/64、γ0.5/0.75、完整QKV逐分支精确Top10%、β=min[0.5(1+R_old),1]、掩码正则0。α按冲突门后的区域范数、ρ=1计算，零BA初始化值0.5；固定字段0.5不是后续实际保护强度。

3090顺序R0 W_pre谱位置、R1任务起点谱位置、R2置换任务起点位置；三组均幅度排名。5090顺序S0幅度、S1 W_pre乘积、S2任务起点乘积、S3置换任务起点乘积；四组均W_pre保护位置。每组执行任务开始评分探针，任务内评分固定；新位置沿用W_pre逐Q/K/V保护数量，而非新评分质量覆盖90%。

## 验证与更新

- 本地16项新专项通过；全量873项检查、15项CUDA相关跳过，其余通过。初次系统Python检查3项缺easydict，使用仓库既有.venv后通过，未以代码改动绕过。
- 独立审查发现并修复汇总顺序问题：先核验任务起点记录和指纹，再发布归因；绘图异常不会留下未经核验的配对。
- 服务器3090全量864项通过；5090首次全量因旧脚本调用系统Python缺numpy、可选离线绘图缺sklearn及历史JSON CA10/测试预期CA5失败。使用指定环境PATH后适用862项通过；明确排除test_baseline_suite.BaselineSuiteTests.test_exact_requested_matrix_and_dataset_recipes、test_feature_geometry.GeometryTests.test_report_end_to_end_with_synthetic_data。本轮七组入口显式CA5，原JSON不覆盖。错误日志和适用测试日志均保留。
- 两机从2832659安全快进相同已推送提交，五份本地JSON哈希前后不变；使用提交bundle传输，未修改Git认证或创建服务器worktree。服务器源码分别核验，不跨机器比较绝对分数。
- 启动前两机均无匹配训练队列，直接启动；不开定时任务、不复用历史PID。若后续需要等待，重新核实外层队列身份再使用Fish lwait。

## 真实运行目录

| 机器 | 外层PID | 原项目 | 本次目录 | 截止启动新组（北京时间） |
|---|---:|---|---|---|
|3090 Ti|1083171|/home/shengqin/lys/baseline/LoDA_ICML2026|logs/shell_logs/task_start_prior_3090/20261012_095303_924576|2026-10-12 17:53:04|
|5090 D|3155562|/mnt/disk1/lys/CIL/code/baseline/lora_base|logs/shell_logs/task_start_prior_5090/20261012_095531_311938|2026-10-12 17:55:31|

总日志分别logs/10_12_task_start_prior_3090_56cb2fd.log与logs/10_12_task_start_prior_5090_56cb2fd.log。每台全部预排配置Task0–1各一轮GPU短测成功后才开始正式T10；首个短测即开始8小时窗口，已启动组完整跑完。训练失败暂停、不自动重试；分析错误记录并保留后续已排训练。

2026-10-12 10:03核验：3090三组、5090四组短测全部退出0、指标有限、Task0–1完整，每组Task1十二层任务起点记录及24条预算记录齐全，保护数量匹配，配置/源码匹配，预算残差最大3.73e-9、P合并诊断误差为0。全模型合并前后最大logit变化不超过3.70e-6，预测变化均为0。3090已进入R0正式T10，训练PID1093001，父PID1083171；5090进入S0正式T10，训练PID3164394，父PID3155562。两机无分析异常、无训练错误。本段为启动进度快照，不能当作正式性能结论。

诊断比较的是任务起点掩码与其置换掩码，在每层每分支完整QKV下匹配同状态有效范数；不保证合成S+P范数或独立训练轨迹等量。旧字段wpre_margin按reference_position标签解释。测试标签仅用于只读诊断，CA仍保存类别协方差。最终成绩、逐任务曲线和成本等待全部T10结束后核验归档。

短测原始日志/run.json、manifest/active/queue/截止记录、两机正式首组部分日志及测试日志保存在[启动核验归档](raw/experiments/2026.10.12_任务起点权重_DM-B-261012-001/启动核验/)，[smoke_health.json](raw/experiments/2026.10.12_任务起点权重_DM-B-261012-001/启动核验/smoke_health.json)保存七组独立核验。总日志部分快照已回填到实验卡的启动台记录，自动跳过短测；不将导入状态当作T10完成证据。
