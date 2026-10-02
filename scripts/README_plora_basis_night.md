# 冻结 P-A 方向来源：3090 夜间六组补充实验

当前两组 gradient、weight_prior 完整T10先跑完，再追加本队列。
不因前三任务结果为负而提前删去夜间候选，但不把候选或短测写成已有效的方法。
只用ImageNet-R seed1993，原项目、原分支、anchor2.5、20epochs、CA5、math-SDPA。
无随机A完整基线重跑、无回放、无测试标签参与训练或路由、无checkpoint保存。

## 为什么补这六组

上一轮Task0–2：分类梯度A增加New约3.30pp，Old下降3.49pp；联合先验New增加3.63pp，Old下降4.09pp。
联合先验没有保住Old，也没有超过分类梯度组。A Gram保持，但有效P更新范数约为随机组的两倍。
因此本轮不宣称已替代历史二阶矩，补验以下三个尚未拆开的问题：
W_pre与累计变化R各自是否有用；保护是否应避开还是利用预训练强响应；
梯度候选只重排是否过窄，以及随机新分类头提供的梯度是否缺乏类别语义。

| 顺序 | 名称 | 只改变什么 | 主要对照／回答的问题 |
| --- | --- | --- | --- |
| 1 | wpre_prior | 只保留W_pre方向能量惩罚 | 对gradient：预训练先验单独是否有用 |
| 2 | history_prior | 只保留R方向能量惩罚 | 对gradient：已学权重变化单独是否有用 |
| 3 | wpre_reuse | 梯度分数偏好W_pre强响应方向 | 对wpre_prior：利用与避让两种假设哪种更合理 |
| 4 | weight_whiten | 联合权重度量允许旋转梯度方向 | 对weight_prior：问题是否在只对固定候选排序 |
| 5 | prototype_gradient | 临时当前训练原型头提供分类梯度 | 对gradient：有类别语义的探测头是否更有用 |
| 6 | prototype_prior | 相同临时头＋原联合先验 | 对prototype_gradient：先验是否仍有额外价值 |

所有新组固定rank、原A完整Gram、B零初始化、A冻结，S、可塑区、双门、正则、学习率、CA及正式推理不动。
不改变冲突坐标评分；也不在夜间按测试成绩自动选下一组或挑系数。

## 具体规则

G：4个当前训练批次（最多192张确定性视图）的分类梯度，限制在原P可塑区。
v/s：G的右奇异方向与奇异值；d：输入维度768。
e_pre = d × ||W_pre v||² / ||W_pre||²；e_old = d × ||R v||² / ||R||²。

```text
gradient       s²
wpre_prior     s² / (1 + e_pre)
history_prior  s² / (1 + e_old)
weight_prior   s² / (1 + e_pre + e_old)       已在前一队列运行
wpre_reuse     s² × (1 + e_pre)
```

weight_whiten不只给上述v排序：临时构造
M = I + d × W_preᵀW_pre / ||W_pre||² + d × RᵀR / ||R||²，
对G M^(-1/2)分解，将所选右方向乘M^(-1/2)映回输入空间，再正交化其span。
这是分类梯度能量相对于权重度量的广义特征方向，不是历史输入协方差的估计或已证明安全子空间。
重构A仍沿用原A的左因子和奇异值，保留完整A Aᵀ。
临时权重Gram/分解张量用后释放，不写入文件，不逐任务累积；CA原有类别协方差仍保留。

prototype两组使用同一批当前训练图片与标签，累计归一化当前模型特征的类别均值，
临时将新头的方向设为均值、保留原头每行范数；缺失类别保持原行。
只通过临时头计算G；随后严格恢复原头、模型模式、梯度标志和Python/NumPy/PyTorch随机状态。
不预训练实际分类头，不将原型保存给推理，也不保留图片/逐样本特征。
与之前head-start区别：正式训练的头仍从原随机参数开始，改变的仅是P-A使用的探测梯度。

## 运行、时间与证据边界

```bash
bash scripts/10_03_imgr10_plora_basis_night_3090.sh --mode dry-run
bash scripts/10_03_imgr10_plora_basis_night_3090.sh
```

默认先六组Task0–1各1epoch短测，全部成功后顺序六组正式Task0–9。
仅运行短测：加`--mode smoke`；已确认短测通过后正式运行：加`--mode t10`。
短测不筛性能，只查路径、有限指标、原头恢复、A冻结/Gram及真实merge更新日志。
根据原3090完整T10每组75–80分钟，追加六组约7.5–8小时，短测约11–15分钟；
权重度量分解有额外开销。含已经启动的两组，总计约10–11小时，不因到点杀掉训练。

本队列输出：`logs/shell_logs/imgr10_plora_basis_night_t10_3090/<timestamp>/`。
每组保存training.log、run.json（实际配置、提交、源哈希），队列保存manifest/queue/results JSON/CSV及标量诊断CSV。
不保存权重。临时头探测准确率/梯度能量、先验重叠降低均只证明代理行为，不证明Old保护。

性能判定：同机同seed看完整Average、Last、Task1–9 Old/New及Forgetting。
New提高但Old下降且总成绩不升，不替换基线；先验组不优于对应gradient，不宣称先验贡献。
当前两组与新组跨提交，但原gradient/weight_prior公式、默认探测头及正式训练设置不改；
保存哈希和相应差异，不称同提交配对。已有随机完整基线仅作历史参考。
方向变化本身会改变学到的更新量，本轮没有严格匹配最终有效更新范数，不能宣称纯方向因果收益。

服务器接续入口`after_plora_basis_night_3090.sh`先等原PID/启动时间对应的两组队列结束，
核对results.json中两组均完成T10且退出码0，才快进原项目、运行专项测试和本队列。
等待脚本复制到项目外运行；不改正在训练的源码，不强制覆盖本机JSON或未提交改动。
