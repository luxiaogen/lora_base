# M90 上恢复原 NCM 三控制量：3090 单组

## 假设与边界

用户授权：以 M90/rank64 为起点，恢复原 NCM 控制的保护覆盖、P rank、保护强度。
这是三个控制量的组合恢复，不是三个独立消融；不恢复乘积排名、能量选点数量或掩码正则。
不修改训练源码，不增加模型接口，不运行 5090，不新设定时任务。

## 配置

- ImageNet-R T10、seed1993，20 epochs、CA5、math-SDPA、Task0 unmasked/anchor2.5；不保存权重。
- S rank64，P 由原控制器决定；S/P gamma=0.5/0.75，非对称权限。
- 原 W_pre 谱保护位置；固定完整层 QKV 10% 幅度选点、冲突强度0.5，step，掩码正则0。
- 仅将 `dual_mask_fixed_coverage=null`、`dual_mask_fixed_protect_strength=null`、`dual_mask_private_rank=0`。
- 原型样本分配关闭。保留上一轮 R0 相同的只读原型位置探针，以免改变诊断协议；评分不用于选位置。

原控制输入不是单独的准确率：C_control=C_new*(1-D_t)，D_t 来自当前训练数据的 NCM 损失诊断。
恢复的公式：coverage=0.70+0.25*C_control，protect_strength=C_control，
P rank=max(1,round(64*(1-0.75*C_control)))。Task0 沿用原初始化和训练路径。
覆盖是重要性质量，不是坐标比例。

## 冲突强度的实际语义

M/M90 及本次的 `dual_mask_fixed_conflict_strength=0.5` 会提前返回，因此虽记录 R_old，
它不控制实际冲突门。原完整方法固定字段为 null 时才使用 min(0.5*(1+R_old),1)。
R_old=max(0,C_new-C_all)，比较的是当前图片在当前类别集与已见类别集下的 NCM 准确率，
并非旧类准确率或真实遗忘。不要从控制器建议日志推断固定覆盖后实际使用了自适应值。

## 执行和配对

入口：`bash scripts/10_10_imgr10_ncm_controller_restore_3090.sh`。
复用队列：仅 NCM_seed1993；Task0–1 各一轮 GPU 短测成功后进入正式 T10。
短测或训练失败暂停，不自动重试；分析失败仅记录。3 小时是新组启动上限，已经启动的 T10 完整结束。

最近健康的同机参照为
`logs/shell_logs/imgr10_prototype_position/20261010_015004_106756/R0_seed1993`，
提交 `87dc9d58f1c222c18870318eb16bb6df11f9121b`。
其 Average=87.154，Last=82.47，最终 Old/New=81.92/87.56，Forgetting=6.1256。
正式比较前核对训练源码哈希、有效配置、数据、软件及硬件；新增队列与分析提交差异应披露。
不把旧单组 M90 队列中带 runtime_error 的记录作为健康严格配对。

## 验证与判断

测试覆盖三个字段差异、原控制公式、固定冲突优先于 R_old、Task0 RNG/前向/梯度、单组汇总。
记录实际各任务 coverage/protect/P rank、更新范数、完整 T10 指标、时间和显存。
单 seed 只能判断组合恢复是否值得进一步确认，不能拆分各控制量贡献或替换正式基线。
本机验证：4 项新增专项测试通过；全量 unittest 806 项，跳过15项，其余通过；Shell 语法、编译和 dry-run 通过。
独立代码审查无阻断项。状态：待远程短测及正式启动核验；结果不预填。
