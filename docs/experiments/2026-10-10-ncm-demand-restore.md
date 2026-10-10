# D_t是否值得保留：3090单变量对照

只将上轮OVERLAP组的保护强度从C_new恢复为C_new*(1-D_t)。
覆盖重要性质量0.90、S/P rank64/64固定；冲突强度仍为min(0.5*(1+R_old),1)。
ImageNet-R T10、seed1993、20epochs、CA5、math-SDPA；Task0 unmasked/anchor2.5。
原谱保护位置、非对称权限、gamma0.5/0.75、全层QKV幅度精确Top10%、step、掩码正则0，
不保存checkpoint；保留同一训练来源原型与只读位置诊断流程。

复用既有开关`dual_mask_ncm_direct_strengths=false`，恢复增量阶段D_t计算。
两个固定强度字段为null，固定覆盖和P rank仍覆盖原控制器建议值。
不修改模型、训练方法或分类目标；不读取旧图片训练或测试标签控制。

## 配对参照

已完整成功的OVERLAP：提交6e3abdd4336d1aaee1e2df86a0ce3e2cb9f37f55，
`logs/shell_logs/imgr10_ncm_overlap_strengths_3090/20261010_144150_947509/OVERLAP_seed1993`。
Average87.167、Last82.47、Old82.18、New85.15、Forgetting5.1600，Task097.10。
StageOld86.4744、StageNew85.8378，完整200个训练epoch，健康。

有效训练配置只改变D_t控制开关；另有prefix/wandb_group与队列/分析提交差异。
实施后核对训练源码哈希、软件、硬件和配置，注明跨提交、训练源码一致。
单seed成绩接近只支持精简候选，不宣布等效、新颖或自动更换基线。
若D_t明显改善Old–New权衡，则保留或推迟删除。

## 执行

入口：`bash scripts/10_10_imgr10_ncm_demand_restore_3090.sh`。
只跑DEMAND_seed1993，Task0–1各一轮GPU短测成功后执行完整T10。
3小时为新组启动上限；短测或训练失败暂停，不自动重试；分析错误单独保存。
核实3090空闲后在原目录更新，保留服务器五份数据JSON；不安排定时任务。

本地3项新增专项测试通过；全量轻量测试818项，跳过15项，其余通过。
独立代码审查无阻断；编译、Bash语法、单组配置及dry-run通过。
