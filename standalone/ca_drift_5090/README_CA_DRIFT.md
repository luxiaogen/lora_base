# 5090：漂移定向的CA特征增广

本目录是独立实验包。冻结已归档的调试B快照，不依赖或修改3090调试目录。
基线来源及原始文件SHA见 `BASELINE_SNAPSHOT.json`，实施记录见 `IMPLEMENTATION.md`。
原README仅保留历史说明；本轮运行规范以本文件为准。

## 固定四组

| 顺序 | mode | 旧类协方差增量 |
|---|---|---|
| A | baseline | 不加；仍执行相同探针 |
| B | full | 当前图片配对漂移的无偏协方差C |
| C | isotropic | trace(C)/768 × I |
| D | permuted | 固定坐标置换 P C Pᵀ |

旧均值均不变。只在Task1–9执行；旧cov每任务累加一次，新类仍用原始统计提取。
取当前训练集的确定性视图，在任务训练前及训练合并后各提取一次当前编码器特征。
不取W_pre特征、不读旧图片或测试标签、不增加损失或可训练参数。
每轮增量任务仅新增一行 `CADriftAudit`，详细数据写入result.json。

共同设置：ImageNet-R T10、seed1993、20 epochs、CA5、math-SDPA、Task0 anchor2.5。
全谱W_pre评分、数量校准的分层覆盖、alpha固定0.5、S/P64/64、Task0 P初始化64，
完整QKV精确Top10%幅度门、beta=min(0.5*(1+R_old),1)、gamma0.5/0.75。
不保存checkpoint；不是旧P39/SVD95 B组的严格复现。

## 5090运行

```bash
LODA_PYTHON=/mnt/disk1/envs/lys_cil/bin/python bash scripts/run_ca_drift_5090.sh \
  --data-path /mnt/disk1/lys/imagenet-r \
  --root /绝对路径/全新运行目录 --hours 10
```

四组全部Task0–1各一轮、CA1短测通过后，自动按A→B→C→D完整训练。
完整组中途不因成绩改变顺序。10小时后不启动新完整组，已启动组跑完。
短测/训练异常暂停且不重试；指标汇总或CSV错误单独记录。
完成性无法确认时不接跑。成功组续跑前核对源码、有效配置与GPU记录。

```bash
LODA_PYTHON=/mnt/disk1/envs/lys_cil/bin/python bash scripts/run_ca_drift_5090.sh \
  --root /同一运行目录 --resume
```

无需定时任务。启动进程后查看 `active.json` 的真实queue_pid/child_pid；
`queue.json`保存状态，`summary.json/csv`保存完整组指标及逐任务审计。

## 验证与解释

```bash
python -m unittest discover -s test -v
```

这是协方差膨胀的增广候选，不是旧类真实统计运输或已校准不确定性。
B仅优于A不能排除普通扩噪；B同时优于C/D才支持方向对应关系有价值。
单seed只作方向性证据；缺失配对不归因，不自动替换正式基线。
