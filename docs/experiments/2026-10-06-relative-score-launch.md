# 相对幅度评分：双机启动记录

实验卡：[DM-B-261006-003](实验日志/DualMask/相对幅度评分/DM-B-261006-003.md)。

执行分支 `codex/mask-budget-comparison-20260924`，训练提交：
`c7ce889fe6702567705d2fcc75bc58c4bbd758e3`，已推送到 GitHub。

两机顺序均为 `wpre_relative` → `task_relative`；各两个完整 ImageNet-R T10，seed1993、anchor2.5、20epochs、CA5、math-SDPA、无权重保存。两种配置先各完成Task0–1一轮GPU短测，全部成功才进入正式训练。

| 机器 | 队列PID | 首个正式训练PID（14:11核验） | 启动时间（北京时间） | 原项目 |
|---|---:|---:|---|---|
| 3090 Ti | 928154 | 935215 | 2026-10-06 14:07:02 | `/home/shengqin/lys/baseline/LoDA_ICML2026` |
| 5090 D | 3088721 | 3093257 | 2026-10-06 14:05:50 | `/mnt/disk1/lys/CIL/code/baseline/lora_base` |

队列主日志：

- 3090：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/10_06_imgr10_relative_score_3090_20261006_140702.log`
- 5090：`/mnt/disk1/lys/CIL/code/baseline/lora_base/logs/10_06_imgr10_relative_score_5090_20261006_140550.log`

运行目录：

- 3090：`logs/shell_logs/imgr10_relative_score_3090/20261006_140702_629907`
- 5090：`logs/shell_logs/imgr10_relative_score_5090/20261006_140550_500249`

目录中 `active.json` 给出当时实际训练PID、阶段和日志；PID会随候选切换，以上队列PID不是永久有效的等待目标。查看进程后再使用 lwait。

## 启动前核验

- 最终本地测试集运行687项，14项跳过，其余通过。两机各13项专项测试通过。代码审查通过。
- 两机历史M参照核对 `matching_issues=[]`；分别复用3090 R0与5090 S2。
- 两机显卡UUID、软件和本机数据路径与对应归档匹配。使用跨提交、默认路径回归核验；不称同提交或全训练源码一致。
- 3090通过已推送提交的Git bundle快进更新，因其GitHub SSH公钥认证失败；未更改认证配置。
- 3090 `imgr10.json` SHA256：`fac0d1e7c6fb383d22d98f110d9e024c679daa2febf3873af6007987c29f30b9`。
- 5090 `imgr10.json` SHA256：`346c0a69e10628c9d2d3b7e8d149c62e4290f32b59eb578cab52fe80e444720a`。
- 快进更新前后，上述配置哈希不变；5090五个本地JSON修改保留。3090已有 `LoDA_dualmask_baseline/` 目录未动。

两台各两个GPU短测均完成、退出码0：3090分别2.036/2.067分钟，5090分别1.758/1.792分钟。14:11核验两台均进入 `wpre_relative` 正式T10，进程实际占用GPU，之后自动接续 `task_relative`。
短测成绩只用于运行健康，不用于性能判断；完整T10结束后由队列生成结果、逐任务曲线和Old–New图。本文件的启动状态不代表候选已完成或已有效。
