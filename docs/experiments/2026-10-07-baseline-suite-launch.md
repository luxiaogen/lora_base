# 原完整 DualMask 四数据集三 seed：3090 启动记录

实验卡：`DM-B-261006-004`。用户选择原完整 DualMask，anchor=2.5，并明确要求 CIFAR-100 也开启 CA；不是 M 配方。

## 固定队列

CIFAR-100 → ImageNet-A → CUB-200 → ImageNet-R；各 seed1993/1996/1997，12组完整T10。
各数据集原学习率、rank、margin、γ不变；20epochs、CA5、math-SDPA、不保存checkpoint。CIFAR的CA开关与anchor仅由命令覆盖，服务器JSON未改。
无时间截止。每个数据集先跑Task0–1、一轮GPU短测，四次全部通过再进入正式训练；短测不用于评价准确率。

## 提交与实际启动

- 分支：`codex/mask-budget-comparison-20260924`
- 训练提交（已推送GitHub）：`497bd4b50b1f8391f47b452dd67e380c1a3c6b6e`
- 服务器原目录：`/home/shengqin/lys/baseline/LoDA_ICML2026`
- 环境：`/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python`，Python3.9
- 显卡：RTX3090Ti，`GPU-980ce4a3-6c17-b57d-ff2d-e43dbcd5c84a`
- 启动：2026-10-07 00:45:56，北京时间；独立会话运行，SSH断开不影响队列
- 队列PID：`1888527`
- 首次CIFAR短测PID：`1888607`（短测，不是正式训练）
- 总日志：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/10_06_baseline_4datasets_3090_20261007_004556.log`
- 运行目录：`/home/shengqin/lys/baseline/LoDA_ICML2026/logs/shell_logs/baseline_4datasets_3090/20261007_004556_786646/`

服务器GitHub SSH认证原先不可用，因此上传已推送提交的增量Git bundle，验证后fast-forward到上述SHA；未改认证、未新建worktree、未覆盖本机JSON或`LoDA_dualmask_baseline/`。
本地9项专项、696项回归（14 skipped）通过；服务器20项专项通过，代码审查通过。

## 数据与配置留存

数据预检（未重分数据或下载）：CIFAR train/test=50000/10000；ImageNet-A=5981/1519；CUB=9430/2358；ImageNet-R=24000/6000。后三者train/test类别映射一致，各200类。这是现有服务器数据划分，不据此称为所有数据集的官方标准划分。

更新前后SHA256相同：

| 服务器配置 | SHA256 |
|---|---|
| cifar10.json | `960b2eb6daae80036d87e72cfcf8c1c7099cf1eb5be76878602fd5dabcd9c0d6` |
| imga10.json | `cd0a111da220356dd997b4a5910f0138eece210ea0229aa8d78bc53d099873e2` |
| cub10.json | `586f95eb2c332432be309defd62c0da35bef6beb165904ce0f3d478f52d69ce1` |
| imgr10.json | `fac0d1e7c6fb383d22d98f110d9e024c679daa2febf3873af6007987c29f30b9` |

各组run.json另存命令覆盖后的有效配置、训练源码/配置哈希、软件与显卡。结果按数据集分别汇总三个seed；未凑齐不报告三seed均值。纯分析错误留存而不中断后续训练；训练异常暂停。

## 查看与续跑

在服务器原项目目录查看总日志：

```fish
tail -n 40 ./logs/10_06_baseline_4datasets_3090_20261007_004556.log
cat ./logs/shell_logs/baseline_4datasets_3090/20261007_004556_786646/active.json
```

需要续跑时，先确认原队列已结束，并保持训练提交和JSON不变；不要与活跃队列重复启动：

```fish
bash scripts/10_06_baseline_4datasets_3090.sh --resume ./logs/shell_logs/baseline_4datasets_3090/20261007_004556_786646
```

续跑校验指纹后跳过成功完整组；失败组保留证据、不会自动重试。

## 2026-10-07 06:45左右的实时核验

四次短测全部完成、退出码0。首组正式CIFAR训练PID `1898936`，有效配置核对为max_tasks10、init_epoch/epochs20、CA=true/CA5、anchor2.5、原乘积排名、掩码正则0.01、save_task_weights=false。

| CIFAR seed | 状态 | Average | Last | Old | New | Forgetting | 分钟 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1993 | 完整T10，退出码0 | 94.846 | 91.74 | 91.73 | 91.80 | 2.9444 | 142.43 |
| 1996 | 完整T10，退出码0 | 94.920 | 91.65 | 91.54 | 92.60 | 3.6778 | 142.43 |
| 1997 | 运行中，Task5 | — | — | — | — | — | — |

当前队列PID `1888527`，训练PID `2139642`，GPU进程已核验。尚有9组其他数据集待启动。运行中的组在queue.json完成前仍显示pending，以active.json确认活动状态，不能将它误读为没有启动。
三seed未齐，不报告均值；这是基线测量，不是提点对照。CIFAR实测每组约2小时22分，12组不会在一夜内全部结束，无时间截止继续执行。纯只读分析错误文件当前未出现。
