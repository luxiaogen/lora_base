# 3090 保护位置 × 冲突排名：四组归因

问题：W_pre是否选中了有价值的保护位置？乘积冲突排名是否在这个框架内额外有效？

| 顺序 | 保护位置 | S/P冲突排名 |
|---|---|---|
| A | 原W_pre图 | 重要性×更新幅度 |
| B | 原W_pre图 | 更新绝对幅度 |
| C | 固定置换图 | 重要性×更新幅度 |
| D | 与C相同置换 | 更新绝对幅度 |

ImageNet-R完整T10、seed1993、anchor2.5、20epochs、CA5、math-SDPA。
全部候选A初始化、方向修正、教师、额外CA修补关闭；保存权重false。
旧基线缺少同源码、同诊断指纹，本轮A只补一次；不追加基线重复或分数扫描。
本机数据路径与device来自exps/dlora/imgr10.json。

`dual_mask_protect_position=wpre`为默认原行为，`permuted`仅从Task1开始置换保护图。
每层Q/K/V分别做行、列置换；保留总数、行占用分布和列占用分布，不保留某一固定行/列原有占用。
私有CPU generator固定seed1993和层/投影偏移，每个任务使用同一映射。P仍为补集；重要性张量不置换。
因此本轮是保护位置在现有框架内的对照，不是所有W_pre信息的去除实验。
本轮统一`dual_mask_conflict_reg_original_score=true`，冲突正则始终按原乘积分数加权。
既有幅度模式原来还会改变正则权重；本轮显式固定这一项以隔离排名因素，默认旧行为不变。

## 运行

```bash
bash scripts/10_03_imgr10_protect_position_3090.sh
# 四组短测：
bash scripts/10_03_imgr10_protect_position_3090.sh --mode smoke
# 仅展开命令，不训练、不写输出：
bash scripts/10_03_imgr10_protect_position_3090.sh --mode dry-run
```

默认四组各先Task0–1×1epoch短测，短测开启三阶段评估检查合并前后变化；正式组不增加这一诊断。
短测成功后A→B→C→D四次完整T10。
短测退出码用于运行检查，不按短测性能筛选。正式负结果也不会跳过后续组。
如训练进程报错，保留已有输出并停止队列；不会覆盖旧目录。
预计约5小时正式训练，加短测约10分钟；按最近3090每组约74分钟估算，实际GPU负载会影响时间。

## 自动产出

目录：`logs/shell_logs/imgr10_protect_position_3090/<timestamp>/`。
每组保存run.json、training.log；主目录保存manifest.json、queue.json。
run.json含有效配置、提交、models/methods/utils与训练入口的源码SHA、Python/依赖版本、实际主机/GPU身份与阶段标识。

每组完成后自动更新：

- results.csv/json：Average、Last、最终Old/New、Task1–9阶段平均Old/New、Forgetting、完成任务数。
- mask_diagnostics.csv：各任务/层保护数量、相对原图Jaccard、保护的重要性质量与P秩。
- update_diagnostics.csv：读取真正合并的raw/safe更新；按任务/层/分支/QKV记录范数、移除量与密度。
- per_task.csv、report.md：逐任务成绩与边界说明。
- 四组指标、120个任务/层保护图及684个任务/层/分支/QKV更新记录完整，Task0相同、指纹匹配且无Traceback后，才生成factor_effects.json中的效应与old_new.png/pdf。

```bash
python3 scripts/analyze_protect_position.py logs/shell_logs/imgr10_protect_position_3090/<timestamp>
```

总移除量定义为`||raw−effective||`；冲突额外移除量为`||pre_conflict−effective||`。
它们不等于`1−||effective||/||raw||`。各分支分别报告，不把S/P相消后的总范数冒充单分支抑制量。
merge_error只是同一状态下重算safe_delta的偏差；实际前后预测一致由独立测试/短测诊断检查，不能混称。
相同保护数不保证相同抑制强度，若两者共同变化，不能只归因为位置更精准。

设置检查只在test中进行；训练路径未加入新增断言或ValueError。
方法定义与证据表见docs/experiments/2026-10-03-dualmask-method-spine.md和2026-10-03-dualmask-evidence.md。
