# 5090：漂移定向CA增广（执行记录）

## 范围与身份

用户授权在5090实现并运行；3090正在调试，本轮未连接、不改动。
本轮四组均重新运行，不能将旧分支成绩作为严格配对。

- 分支：`xiaogen/ca-drift-5090-20261014`。
- 固定训练提交：`83cfd20d3982941db1328d12fca33e0b92b627df`。
- 代码：`standalone/ca_drift_5090`，冻结此前已归档的调试B快照。
- 归档SHA256：`b19e12dc7f02390735dfdc3d55b8e1538f6ce7af8d1bf32f5ec4882751774d2b`。
- 5090独立目录：`/mnt/disk1/lys/CIL/code/baseline/LoDA_ca_drift_5090_1014.KH4lDG`。
- 运行目录：上述目录下`runs/ca_drift_seed1993_83cfd20`。
- SSH别名：`5090d`；解释器：`/mnt/disk1/envs/lys_cil/bin/python`。
- GPU：RTX5090 D，UUID `GPU-321e3949-0a78-d0d8-39cb-9ac05b0fe67a`。
- 数据：`/mnt/disk1/lys/imagenet-r`；torch2.7.1+cu128、torchvision0.22.1+cu128、timm0.6.12。

通过git归档部署，不是服务器git checkout。提交对应关系保存在`deployment.txt`，每组`run_info.json`和队列另保存源码逐文件SHA，不能把git_commit为空误报为同提交checkout。

## 固定配方

ImageNet-R T10、seed1993，20epochs、CA5、math-SDPA、Task0 anchor2.5、不保存checkpoint。
全谱W_pre评分、数量校准分层重要性质量覆盖、alpha固定0.5、S/P64/64、Task0 P初始化64；完整QKV精确Top10%幅度门，beta=min(0.5*(1+R_old),1)、gamma0.5/0.75。
这不是旧P39/SVD95的B组复现。骨干、数据、分类损失等10份冻结文件SHA未改变。

## 四组与解释

Task1–9在训练前及合并后，对同一批当前训练集确定性视图提取当前编码器特征，取配对差d。旧类均值不变，旧cov每任务至多累加一次，当前新类仍使用原统计计算。

| 顺序 | mode | 旧cov增量 |
|---|---|---|
| A | baseline | 零，执行相同探针但保持原cov对象 |
| B | full | C=Cov(d)，逐坐标中心化、N−1分母 |
| C | isotropic | trace(C)/768 × I |
| D | permuted | P C Pᵀ，独立RNG固定坐标置换 |

无旧图片训练，无新增损失、教师或可训练模块。该操作属于漂移驱动高斯伪特征增广，不是精确统计运输或已校准不确定性。B优于A不足以排除普通扩噪；需结合C/D与Old/New判断方向对应关系的价值。单seed不作等效、显著性或新颖性证明，不自动替换基线。

## 验证与运行策略

本地32项测试：31通过、1项CUDA跳过；5090最终32项全部通过，含真实CUDA RNG恢复；Python编译和Bash语法通过。独立代码审查完成；两处续跑问题已先补失败测试再修复。

四组全部Task0–1各一轮、CA1短测成功后，固定A→B→C→D进行正式T10。10小时停止启动新完整组，已启动组跑完。短测或训练失败暂停、不重试；分析错误独立记录、不阻止已核验成功后续组；不设置定时任务。

队列PID首次核对：`525123`。当前状态以`active.json`、`queue.json`和实际`ps`为准，不复用历史训练子PID。
总日志：独立代码目录下`queue_console.log`。每组日志为运行目录下`smoke|full/组名/console.log`；原始训练证据位于各自`training`目录。
最终指标汇总到运行目录的`summary.json`和`summary.csv`，保存Average、Last、最终/阶段Old/New、Forgetting与逐任务CADriftAudit。

尚在执行，本记录不预填准确率或完成结论。
