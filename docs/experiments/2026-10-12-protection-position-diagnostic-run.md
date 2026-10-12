# 保护位置诊断：3090运行记录

训练代码已提交并推送至研究分支 `codex/mask-budget-comparison-20260924`：
`920418a42bdacd74c23ad39d4edd95f65d5e592e`。

本地17项专项测试通过；全量905项通过、跳过15项。服务器896项轻量测试全部通过。独立审查发现的诊断完整性、短测／正式指纹逐值核对、正式启动前指纹门禁均已修复并复核。两个训练文件只增加只读钩子，不修改默认前向或最终合并公式。

上一轮固定P支持A/B/C/D已核验全部完整T10成功，匹配问题为空。本轮不采用固定P支持，恢复A配方F：ImageNet-R T10、seed1993、W_pre谱重要性质量覆盖0.90、rank64/64、γ0.5/0.75、非对称权限、固定α0.5、精确Top10%幅度门、β=min[0.5(1+R_old),1]、掩码正则0、20epochs、CA5、Task0 unmasked/anchor2.5、math-SDPA，不保存checkpoint。

3090空闲，2026-10-12 20:44:47（北京时间）直接通过既有Fish `lrun` 启动；没有lwait等待项或定时任务。通过已推送提交的增量Git bundle快进原项目，未修改认证设置；五份本机数据JSON SHA256快进前后逐项一致。

- 原项目：`/home/shengqin/lys/baseline/LoDA_ICML2026`。
- 环境：`ICML2026_LoDA`，Python3.9.25、torch2.0.1、torchvision0.15.2、timm0.6.12、numpy1.24.3。
- GPU：RTX3090 Ti，UUID `GPU-980ce4a3-6c17-b57d-ff2d-e43dbcd5c84a`。
- 外层Conda PID：1696004；Python队列PID：1696049。
- 首次GPU短测PID：1696086（历史启动记录，不是正式训练PID）。
- 总日志：`logs/10_12_imgr10_position_signal_3090_920418a.log`。
- 运行目录：`logs/shell_logs/protection_position_diagnostic_3090/20261012_204449_837679`。

20:46:19核验仍在GPU短测Task1第2轮，无错误。Task1采样Fisher已经完成40张旧训练来源图片、每图两次采样；评分集、旧类评估集和新类评估集分别40张，评分与旧评估不重叠。任务起点求导约1.08秒，全部构图约1.27秒；当前任务临时CPU评分／掩码缓存191,102,976字节。这只是短测成本，不能代表Task5/9或完整训练成本。

短测Task0一轮、Task1三轮，诊断第2/3轮；完整性及指纹检查通过才自动进入唯一正式组 `imgr10_F_seed1993`。正式训练始终用原谱位置，Task1/5/9起点构图、第2/20轮只替换S做五位置等范数诊断。P、分类头、β和幅度门固定；旧训练图片只作诊断，不用于学生优化或CA。五位置不是五组训练；MC不是性能上界。

20:49:06核验：短测完整结束、退出0，耗时2.6101分钟；两次S-only诊断分别1.8218/1.8226秒。全部诊断记录及正式启动前源码、硬件、软件指纹门禁通过，唯一正式组已经启动，训练PID1699864。正式日志为上述运行目录中的 `imgr10_F_seed1993/training.log`。

本地重新解析归档的原始短测确认：Prepared/Fisher/Mask/Score/Update/Intervention/Cost记录数分别1/12/60/24/120/24/2，完整性问题为空；短测与正式run.json的源码、硬件、软件指纹差异为空。启动台台账已实际读取实验卡DM-B-261012-003并关联本次入口；没有开启或宣称HTTP服务正在监听。

目前没有正式完整成绩，不启动残差三组、Fisher训练、5090或多seed。普通F历史约65分钟，额外求导和诊断成本以实际日志计，不提前保证结束时刻。启动核验已归档CPU回归、已完成短测原始日志、manifest/active/queue及正式启动run.json；这不是T10完成归档。

[预注册计划](../plans/2026-10-12-protection-position-diagnostic.md)；[入口说明](../../scripts/README_protection_position_diagnostic.md)；[实验卡](实验日志/DualMask/保护位置诊断/DM-B-261012-003.md)。后续运行身份以服务器manifest、active.json、queue.json及ps为准，不能复用此处历史PID。

## 完整结束

21:53正式一组T10成功结束，退出码0，耗时65.8649分钟；十阶段、200个epoch及六状态诊断全部完整。随后核验外层、队列和训练PID均已退出。上述启动时“目前没有正式完整成绩”等是历史快照，不是当前状态。

Average 87.273、Last 82.55、Old 82.11、New 86.70、StageOld 86.497、StageNew 86.494、Forgetting 5.6644。真实残差有旧类margin线索，但新类margin下降、旧类准确率未提高，不采用为有效新组件，不自动追加残差训练。正式日志下载SHA与服务器一致，完整数据、图及快照已归档，见[完成报告](2026-10-12-protection-position-diagnostic-results.md)。
