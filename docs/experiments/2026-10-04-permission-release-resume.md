---
experiment_id: DM-B-261004-002
date: 2026-10-04
status: resumed_A2_formal_training_verified
original_training_commit: e7e9d5c4f2f1022e95c3198fbd2bf2371fa18fed
fix_commit: 990dab84252507df8fdb02181179cc654cc13d7f
resume_machine: RTX3090Ti
resume_groups: [A2, A3, A4, A5, A6]
resume_budget_hours: 6.5
save_task_weights: false
queue_pid_3090: 3072420
initial_formal_training_pid_3090: 3072426
started_at_3090: 2026-10-04 23:30:21 +08:00
---

# 汇总字段修复与3090未完成组续跑

用户重新明确授权修复、测试、推送后立即在3090续跑A2–A6。
原10小时窗口不重新生效；新窗口仅针对这五组，6.5小时后不启动新组，活动组完整结束。
不重跑已经完成的A0/A1，不新增候选、基线或参数扫描，不保存权重。

## 错误与修复范围

原3090在A1完整T10退出0后，后处理触发 `dict() got multiple values for keyword argument 'mode'`。
实验组编号和日志中的选点策略都叫 `mode`。现在实验组保留为 `mode`，选点策略保留为 `release_selection_mode`。
新增 `--modes` 只调度给定实验组；不指定时仍执行原完整队列。跨机器组和重复组在命令入口拒绝。
只修改汇总、调度入口、对应测试和README；模型、训练代码及实验规格不改。

## 验证

- 先运行新增回归测试，复现真实字段冲突和缺少选组入口；修复后8项队列测试通过。
- 本地完整回归628项通过，11项CUDA/环境相关跳过；Python编译和diff空白检查通过。
- 以归档的真实A0/A1训练日志、run.json和queue.json重汇总，完整T10且matching_issues为空。
  Average仍为87.218/87.224，Last仍为82.50/82.47；324条放行记录保留 `benefit` 与 `A1` 两种语义。
- 独立只读代码审查无阻断；没有放宽跨提交或源码指纹检查。
- 原七组GPU短测都已通过，此次没有改GPU计算路径，续跑不再重复短测。
- 3090原Python3.9环境8项测试全部通过，原A0/A1派生汇总恢复成功。
  安全快进到修复提交；五个本机JSON更新前后SHA256均不变；Git对比训练源码/实验规格无变化。
- 已下载并按远端SHA256核对续跑 [manifest](raw/experiments/2026.10.04_有限权限_DM-B-261004-002/resume_3090/manifest.json)
  与 [A2 run.json](raw/experiments/2026.10.04_有限权限_DM-B-261004-002/resume_3090/run.json)。
  两份SHA256分别为 `bd3bd421fd851d56327aff101c7c56fa4ffa3c0d2773de2a3f60ba964c10153c`
  和 `632ed3a3474bff19b2dddd6602f62509f95893c845bfefabe469699549154a4f`。
  A2对原A0的软件/硬件指纹相同；保存的源码差异仅为汇总与选组两个脚本。
  有效配置差异只有预登记的放行策略off→random及运行前缀；T10、seed1993、anchor2.5、20轮、CA5、math-SDPA、不保存权重均核对一致。

## 续跑身份与比较边界

原输出目录：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/shell_logs/imgr10_permission_release_3090/20261004_111724_546550/`。
原始A0/A1日志和run.json不覆盖。原目录派生汇总可用修复脚本重新生成。
续跑到新的时间戳目录，记录真实新提交与配置/源码/硬件指纹。
跨新旧目录时提交及两个脚本哈希不同，自动严格归因不会绕过检查；最终结合完整记录核对训练源码和非目标配置后，披露这项编排修复差异。

已启动队列PID为3072420，固定顺序A2/A3/A4/A5/A6；没有再次启动A0/A1。
主日志：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/shell_logs/10_04_imgr10_permission_release_3090_resume_20261004_233021.log`。
新输出：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/shell_logs/imgr10_permission_release_3090/20261004_233021_034577/`。
启动约23:30:21，不再启动新组的截止约2026-10-05 06:00:21；最后活动组完整结束。
预计五组约6～6.5小时。PID及运行状态是此次核验快照，不冒充未来持续存活状态。
启动50秒后经ps核对队列仍存活，主日志已显示 `Starting A2 REAL FULL T10`。
随后读取子进程/首轮日志时SSH连接超时；这不是训练报错，也不据此再次启动队列。
23:33:14以后通过5090跳板只读复核，active.json显示A2正式训练子进程3072426；
队列ps显示启动3分06秒仍存活，A2训练日志已到Task0第10/20轮，没有把启动状态当作完整结果。
跳板只用于SSH传输，没有在5090启动或重启任何实验。
用户已接受连接不稳定时自行核查，备用只读命令如下；不要直接重跑A2–A6。

```bash
ps -p 3072420 -o pid,ppid,etime,args
cat /home/shengqin/lys/baseline/LoDA_ICML2026/logs/shell_logs/imgr10_permission_release_3090/20261004_233021_034577/active.json
tail -n 20 /home/shengqin/lys/baseline/LoDA_ICML2026/logs/shell_logs/10_04_imgr10_permission_release_3090_resume_20261004_233021.log
```

```bash
/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python scripts/run_permission_release_night.py --machine 3090 --mode t10 --hours 6.5 --modes A2 A3 A4 A5 A6
```

## 5090核对（只读，未续跑）

已恢复SSH读取：原队列记录B0–B4全部退出0；B4结束后的汇总出现同一mode冲突，B5–B11未启动。
原队列JSON记录耗时分别为137.390、87.157、80.425、146.135、124.260分钟。
这次只核对完成状态，没有重新启动5090，也没有把此前连接失败理解为实验失败。
5090的完整训练日志和性能表仍需单独恢复、核对与归档，不能从退出状态推断提点。
