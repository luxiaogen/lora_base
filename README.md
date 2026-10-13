# DualMask：旧 B 的独立精简复现

分支 `xiaogen/depth-b-repro-20261014` 仅保留 ImageNet-R T10、seed1993 的校准分层 B 配方。main 调试目录和研究分支不受影响。其它数据集配置、旧队列、教师、原型路由、区域预算、固定 P 支持、额外掩码正则及 NCM 控制 rank／覆盖／保护强度的路径已删除，不是靠关闭开关隐藏。

## 唯一配方

参考训练提交：`fa53a839d4a8df7edd28afdf230b577b548179b6`。旧 B 的完整曲线为：

```text
97.10, 92.66, 90.30, 88.07, 85.83, 84.66, 84.72, 84.14, 83.01, 82.48
```

参考指标：Average 87.297、Last 82.48、最终 Old/New 82.03/86.70、阶段平均 Old/New 86.4822/86.6667、Forgetting 5.9544。这些是旧实验成绩，不是新代码已经通过完整复跑的声明。

| 项目 | 本分支设置 |
| --- | --- |
| 数据／训练 | ImageNet-R T10，seed1993；每任务20类、20轮，batch48 |
| 优化器／分类 | SGD，lr0.02、momentum0.9、weight decay0；余弦调度；CosFace scale20、margin0.1 |
| 谱评分 | 原 W_pre；SVD最多768方向，保留95%的奇异值平方能量，再按原公式构造重要性 |
| 保护覆盖 | 校准分层质量覆盖；不是坐标比例，也不是精确0.95→0.85 |
| 保护强度 | 所有层固定 α=0.5；本分支不包含分层 α 或预算 α |
| 增量分支 | S/P rank64/64；A冻结、B训练；γ=0.5/0.75 |
| 更新权限 | S在保护区减半；P仅在保护补集生效 |
| 幅度门 | S/P分别精确选择完整packed QKV的Top10%，选中位置乘1−β |
| β | min[0.5×(1＋R_old),1]；R_old=max(0,C_new−C_all) |
| Task0 | S-only、unmasked，S的A/B训练，独立anchor权重2.5 |
| Task0随机兼容 | 仍构造但冻结、不使用P，初始化rank39；此值只为复现旧seed1993随机轨迹 |
| 其它 | 掩码正则0，CA5、CA lr0.01、logit_norm0.1；FP32、math-SDPA；不保存checkpoint |

保护覆盖公式是 `q_l=q_center+0.05*(1−2*l/11)`，l=0…11。Task1只校准一次中心，使总保护坐标数尽量匹配统一0.90重要性质量覆盖。旧 B 中心为0.8944721110165117，浅／深端点约0.944472→0.844472，Task1只比数量参照多2个坐标。本分支按原算法重新计算中心，不把旧中心硬编码进去。

两处覆盖不能混淆：`svd_energy_coverage=0.95`决定保留哪些谱方向；`protect_mass_coverage=0.90`是后续保护坐标数量校准的质量参照。

NCM只计算C_new、C_all及R_old来设β；不计算D_t，不自适应rank、覆盖或α。`task0_p_init_rank=39`显式写在配置中，增量P的rank读取配置的`rank=64`。

## 代码职责

```text
main.py                       单配置入口与参数检查
trainer.py                    随机状态、任务循环、必要日志和结果记录
methods/dlora.py               分类训练、NCM、CA、评估
models/attention.py           初始化、谱权限、幅度门、合并
models/network.py / vit.py     原ViT与已见类别分类头
utils/depth_permissions.py    一次校准分层质量覆盖
utils/data*.py                只读现有train/test，不移动或重切数据
scripts/run_reproduction_3090.sh  回归→短测→一次完整T10
scripts/check_reproduction.py 检查运行健康与旧B指标差异
test/                         配方、数值回归及运行记录测试
```

QR抽样后再Kaiming覆盖、Task0未使用的P、原只读数据遍历仍保留，因为它们会推进随机状态。删除它们就不是相同seed的旧初始化轨迹。CA仍维护类别均值／协方差，不应声称本方法完全没有二阶统计。

日志沿用main格式：`时间 [文件名] => 内容`。每轮一条训练信息，包含Loss、CE、anchor_reg、anchor_reg_weighted与Train_accy；每任务一条NCM与一条权限摘要、CNN结果；最后汇报曲线、Average、Last、Old/New和Forgetting。不开逐步／逐层刷屏日志，完整控制值保存在JSON中。

## 手动运行

使用原3090环境：Python3.9.25、torch2.0.1、torchvision0.15.2、timm0.6.12、numpy1.24.3。独立目录中执行，保留原数据路径：

```sh
python main.py --config exps/dlora/imgr10.json \
  --epochs 1 --ca-epochs 1 --max-tasks 2 --output logs/smoke_B
```

短测通过后完整T10：

```sh
python main.py --config exps/dlora/imgr10.json --output logs/B_seed1993
python scripts/check_reproduction.py logs/B_seed1993
```

输出目录必须未存在；省略`--output`会生成新目录。外部JSON和`--set`都只接受本配方已有字段，已移除的研究开关会在入口报错。支持`--data-path`、`--device`等运行参数；改变算法或训练字段后，不能仍称为旧B复现。

## 固定提交自动验证

以下是Bash脚本，Fish交互终端也用`bash`调用。运行前必须已提交、推送并在独立服务器目录检出该提交；不得在现有运行目录更新代码。

```fish
env B_REPRO_PYTHON=/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python \
  DUALMASK_REFERENCE=/绝对路径/旧B源码归档 \
  EXPECTED_REVISION=实际完整提交SHA \
  PREDECESSOR_RESULT=/绝对路径/前序完整实验/result.json \
  bash scripts/run_reproduction_3090.sh
```

归档用于对照测试，至少包含`models/attention.py`、`models/losses.py`、`methods/dlora.py`、`utils/depth_permissions.py`、`utils/toolkit.py`、`reference_run.json`。未提供归档时，普通单元测试会明确跳过旧源码回归；自动验证入口则拒绝继续。

脚本检查固定提交和干净状态，执行全部CPU/CUDA回归，再Task0–1各1轮短测，健康检查通过后只启动一次完整T10。通过`queue.json`、`active.json`记录真实PID；测试或训练失败立即停止，不自动重试。完整运行后输出`formal/comparison.json`，准确率差异只报告，不用来筛选或重跑。

服务器忙时，在核实当前训练／外层队列PID后，用已有Fish `lwait PID "接续命令"`。`lwait`只表示进程退出；自动验证必须用`PREDECESSOR_RESULT`指定前序完整T10的`result.json`，脚本检查成功状态后才继续。`PREDECESSOR_PID`可额外检查前序PID已经退出。默认任何Python GPU进程驻留都会拒绝接跑，用户明确确认暂停的调试会话才可通过`ALLOW_IDLE_DEBUGGER_PID`逐PID授权，并仍需通过两次空闲检查。不要等待永久存在的交互Shell，也不要终止用户调试会话。

## 验证边界

CPU微型回归比较真实旧源码的初始化/RNG、前向、损失、梯度、优化步、合并及CA；本机无CUDA时两项GPU测试明确跳过，交由3090验证。单元测试不能替代完整T10；最终以同环境运行的配置、源码指纹和10阶段曲线为准，不把一次复跑当作跨seed性能等效证明。

旧NME曾把[N,1]预测与[N]标签广播，可能得到超过100%的诊断值。本分支修正NME报告维度，但保留其数据遍历，正式CNN指标不改变。Task0无效掩码统计也不再沿用自适应控制；Task0前向、anchor与合并仍完全unmasked。
