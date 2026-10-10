# C_new保护＋原R_old冲突强度：3090单组

用户指定冲突强度为min(0.5*(1+R_old),1)，R_old=max(0,C_new-C_all)。
保护强度仍为C_new，Task1起不使用D_t；覆盖质量0.90、S/P rank64/64固定。
其余沿用直接R_old组：ImageNet-R T10 seed1993，20epochs、CA5、math-SDPA，
Task0 unmasked/anchor2.5；谱保护位置、非对称权限、gamma0.5/0.75，
全层QKV幅度精确Top10%、step、掩码正则0、不保存checkpoint。
保留同一只读原型位置探针，正式训练每张图片共同训练S/P，无原型梯度分配。

新增`dual_mask_ncm_conflict_mode=scaled`，默认direct保持上轮候选行为。
新模式沿用已有old_overlap_adaptive公式；两个固定强度字段为null。
所有变化从Task1开始，Task0及原完整方法缺省行为不变。

## 配对

直接R_old参照：提交a40f881b5c9aed82c2c37e29665d3a147fe55958，
`logs/shell_logs/imgr10_ncm_direct_strengths_3090/20261010_131918_840949/DIRECT_seed1993`。
完整健康T10：Average86.618、Last81.17、Old80.32、New89.12、Forgetting9.0644。
本组相对它唯一算法因素是冲突强度规则；另有前缀、W&B组和新增提交差异。
M90参照Average87.154、Last82.47，保护与冲突都固定0.5；与它比较仍是两个强度的组合改变。

源码和配置指纹、硬件/软件需核对；不称严格同提交，不以单seed宣布等效、新颖或更换基线。

## 启动

入口：`bash scripts/10_10_imgr10_ncm_overlap_strengths_3090.sh`。
仅OVERLAP_seed1993；先Task0–1一轮GPU短测，再完整T10。
短测或训练失败暂停，不自动重试；分析错误独立记录。3小时为新组启动上限。
只使用3090原目录，保留本地数据JSON，核对空闲再部署；不安排定时任务。

本地3项新增专项测试及原直接强度6项测试通过；全量815项，跳过15项，其余通过。
独立代码审查无阻断项；编译、Bash语法、单组配置及dry-run通过。
