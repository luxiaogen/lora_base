# 原型分差能否改善保护位置？

本轮只替换 M90/rank64 的保护位置，不加入损失，不做样本梯度分配。
ImageNet-R T10、seed1993、20轮、CA5、math-SDPA、Task0 anchor2.5；不保存权重。
S/P rank64/64、γ0.5/0.75、保护0.5、全层QKV精确10%幅度选点减半、掩码正则0。

| 机器 | 固定顺序 | 同状态范数控制 |
|---|---|---|
| 3090 | R0谱→R1高敏感→R2打乱原型关系→R3低敏感 | 关闭 |
| 5090 | S0谱→S1高敏感→S2打乱关系→S3固定置换谱位置 | 四位置共同最小值 |

各层Q/K/V分别保留原谱90%重要性质量掩码的坐标数，不把新评分的90%质量视为等量。
任务开始用80%原型训练分区的最多128张当前图片，逐样本求分差梯度、平方再平均。
正类为当前训练标签，最近错误类身份固定；全部已见训练来源原型参加比较。
20%独立分区最多64张用于只读敏感度覆盖诊断，不参与原型或位置选择。
真实与无固定点打乱对应评分都计算；参照组也执行相同探针。任务内位置固定。
临时W_pre不更新；评分矩阵用后释放，只保留当前任务布尔掩码，不保存图片、特征或逐样本梯度。

```bash
# Bash脚本；请使用对应服务器的真实训练Python
LODA_PYTHON=/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python \
  bash scripts/10_10_imgr10_prototype_position_3090.sh
LODA_PYTHON=/mnt/disk1/envs/lys_cil/bin/python \
  bash scripts/10_10_imgr10_prototype_position_5090.sh
```

`--mode dry-run`只显示命令；`--modes R2_seed1993 R3_seed1993`选择已预排的剩余组。
`--resume 运行目录`仅限同机器、同提交、同配置及源码指纹，成功组不重跑；失败组不自动重试。
每台先完成四组Task0–1、一轮GPU短测，全部通过才进入完整T10。
首个短测起计8小时，截止后不启动新组，已启动组完整跑完。短测/训练异常暂停该机；
分析异常单独写入analysis_errors.jsonl，不能阻断已排训练。
3090用已核实外层队列PID的Fish lwait接续；等待期间不更新训练目录。
旧整队退出后仍须检查全部T10成功、配置/源码身份；lwait退出不等于实验成功。

运行目录下manifest.json、active.json、queue.json、smoke_queue.json是状态依据。
每组有training.log、run.json配置/源码/硬件软件指纹；汇总生成results、pairs、matching_issues、
tasks、epochs、masks、probes、holdout、diagnostics、norms、costs、storage及配套图。
探针、实际更新和同状态四位置诊断有独立JSON日志标记。

Task1/5/9最后一轮、合并前，用固定最多512旧类/128新类测试图片作四位置等范数诊断。
测试标签只报告全局margin/准确率/预测变化，不选择位置，不调整参数，不改变队列。
同状态范数匹配不保证不同训练轨迹等量；单seed不证明显著或等效；不自动替换正式基线。
R1超过R0只是整体策略收益；超过R2和R3、且S1超过S0/S2/S3，才增强类别关系/方向假设证据。
若低敏感更好，不再声称敏感即应保护；若随机/打乱接近，不宣称语义独立有效。
谱参照也带本轮验证成本；CA仍保留类别协方差，不能宣称整个方法零二阶统计。
