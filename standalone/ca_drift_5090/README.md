# DualMask：main 调试基线

此 `main` 默认使用已确认的 ImageNet-R T10 校准分层 B 配方，额外只保留用户要求的分层保护强度开关。研究分支、旧脚本和删除文件仍可从 Git 历史恢复；不包含教师、原型路由、预算α、固定P支持或其它消融模式。

## 配方

- 参考训练提交：`fa53a839d4a8df7edd28afdf230b577b548179b6`。旧B：Average 87.297、Last 82.48；这是参考成绩，不是本次精简代码已经复跑的成绩。
- Task0：S rank64，A/B均训练，unmasked，独立 anchor2.5。保留原未使用P的初始化随机消耗。
- Task1–9：S/P rank64/64，随机A冻结、B训练，γ=0.5/0.75。
- W_pre谱重要性；`q_l=q_center+0.05*(1-2*l/11)`，浅层覆盖高、深层低。Task1仅校准一次中心，使总保护坐标数尽量匹配统一0.90质量覆盖；旧B中心约0.894472，端点约0.944472→0.844472，不是精确0.95→0.85，也不是坐标比例。
- S保护强度默认固定0.5；`layerwise_alpha=false`。开启时使用 `α_l=0.50+0.25*(1-2*l/11)`，即浅层0.75→深层0.25，只从Task1生效。P始终只允许保护补集。S/P分别按幅度，精确选择完整packed QKV的10%坐标。
- `β=min(0.5*(1+R_old),1)`，`R_old=max(0,C_new-C_all)`；掩码正则0。NCM不再给覆盖或α赋值，也不再计算增量任务rank；仅Task0通过独立的 `set_task0_legacy_rank` 保留未使用P的初始化RNG，后续NCM计算R_old。
- 20 epochs、CA5、batch48、SGD lr0.02、CosFace scale20/margin0.1、CA logit_norm0.1、FP32及math-SDPA、不保存checkpoint。

## 文件职责

```text
main.py                  参数入口；默认exps/dlora/imgr10.json
trainer.py               随机状态、任务循环、日志、结果JSON
methods/dlora.py          分类训练、NCM、CA和评估
models/attention.py      W_pre谱权限、幅度门、初始化、合并
models/network.py        单编码器及已见类别分类头
models/vit.py            原ViT结构与预训练权重加载
utils/depth_permissions.py 校准分层质量覆盖
utils/data*.py           只读取现有train/test；不移动/重切数据
test/                    核心契约与可选旧B数值回归
```

复现需要保留原初始化顺序。被随后Kaiming覆盖的QR抽样、Task0未使用P、ViT兼容参数初始化仍影响RNG，因此没有简单删除。CA仍存类别均值/协方差；不是完全不保存二阶统计。

## 运行

使用原3090环境（Python3.9、torch2.0.1、torchvision0.15.2、timm0.6.12）。本次不替你连接或启动服务器。将此目录复制到新的服务器目录后执行：

```sh
python -m unittest discover -s test -v
python main.py --config exps/dlora/imgr10.json \
  --data-path /home/shengqin/ds_and_pretrain/ds/imagenet-r \
  --epochs 1 --ca-epochs 1 --max-tasks 2 --output logs/debug_smoke
```

短测成功后完整T10：

```sh
python main.py --config exps/dlora/imgr10.json \
  --data-path /home/shengqin/ds_and_pretrain/ds/imagenet-r \
  --output logs/debug_B_seed1993
```

输出目录必须未存在；省略 `--output` 时自动生成新目录。支持原 `--config` 与已有训练字段的 `--set KEY=VALUE`，不接受已经删掉的研究开关。模型身份固定为ImageNet-R T10、rank64/64、ViT-B/16和上述γ；入口拒绝不相符的结构参数，避免配置记录与实际实现不同。默认只跑seed1993，调试时可改学习率等已保留训练参数；改动后的结果不能再声称复现上述B配方。

需要切换到分层保护强度时，显式开启：

```sh
python main.py --config exps/dlora/imgr10.json \
  --set layerwise_alpha=true --output logs/debug_layerwise_alpha_seed1993
```

此时覆盖曲线、rank、P权限和其它训练设置不变，但不再称为旧B；启动日志与 `run_info.json` 会记录分层α。默认关闭时，原B训练路径保持不变。清理后Task0不再构造NCM自适应的无效保护掩码，因此Task0掩码统计与旧记录不同；Task0前向和合并仍unmasked，这不改变训练。

日志沿用 `main`：`时间 [文件名] => 内容`、`Task ..., Epoch ... => Loss ..., Train_accy ...`、`CNN top1 curve`、`Average Accuracy`、`Last Accuracy`。每轮一条，不输出每层掩码或逐步诊断。

完整配置、源码/环境指纹、逐任务结果、控制值、Old/New、阶段均值、Forgetting写入 `effective_config.json`、`run_info.json`、`result.json`。

## 验证边界

`test/test_reference_regression.py` 对照旧B检查真实初始化RNG、前向/损失/梯度、SGD、合并与校准。需用 `DUALMASK_REFERENCE` 指向含 `models/attention.py`、`models/losses.py`、`utils/depth_permissions.py`、`reference_run.json` 的旧B归档；未提供会明确跳过，不当作回归通过。普通核心/入口测试不依赖归档。

旧NME诊断曾将[N,1]预测与[N]标签广播，可能超过100%；本分支仅修正NME报告维度，保留其遍历和RNG，正式CNN指标不变。

单元/CPU回归不代替完整T10性能复现。最终请比较10阶段曲线和全部指标；本次没有运行服务器短测或完整训练。
