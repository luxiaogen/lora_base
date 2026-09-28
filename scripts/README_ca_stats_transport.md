# CA 旧类统计校正：5090 独立可选候选

假设：旧类均值/协方差留在历史骨干的特征坐标中，可能不再匹配当前骨干。
这里只测试一个简单的逐通道仿射校正，不声称已经证明统计漂移是瓶颈。
与3090间隔损失独立，`ca_cross_task_margin_weight=0`。

## 具体方法

Task1–9，在当前任务训练前后，对同一批当前训练图片的固定中心裁剪视图提取特征：

```text
x = 训练前合并骨干产生的特征
y = 训练后合并骨干产生的特征
y ≈ x * scale + offset
旧均值' = 旧均值 * scale + offset
旧协方差' = diag(scale) × 旧协方差 × diag(scale)
```

不需要保留旧模型副本：先缓存x，再正常训练，最后提取y。缓存仅含当前训练特征，
不用旧图片，不保存特征到磁盘，不访问测试/验证集拟合变换。
两次DataLoader使用独立generator、不shuffle；复用的collect_features恢复模型模式及
Python/NumPy/Torch随机状态，不改变训练随机流。

每通道拟合带身份映射先验的最小二乘。设中心化数据为xc/yc：

```text
v = sum(xc², samples)
ridge = mean(v)（数值下限1e-12）
scale = clip((sum(xc*yc) + ridge) / (v + ridge), 0.5, 1.5)
offset = mean(y) - scale * mean(x)
```

拟合在CPU float64上执行，返回统计保持原dtype。正的有界对角缩放保留协方差
正定结构；这是算法约束，不是运行时配置验证。上述范围和ridge规则本轮固定不扫描。
只在生成当前新类统计之前，更新已有旧类统计；新类仍直接用当前特征计算。
已修正的旧统计留给后续任务，每任务累计校正一次。CA原有年龄缩放、每类256条采样、
5轮CE和全部分类头训练不变。Task0不校正。`ca_stats_transport`默认false。

**关键限制：** 新类上的通道变换未必能外推到旧类。日志pair_mse_after下降只是拟合
当前图片更好，不证明旧类分布修正正确，更不证明性能提升。简单对角变换也不能表示旋转。

## 配方与命令

ImageNet-R T10、seed1993、anchor=5、20轮、CA5、math-SDPA。5090维持已有完整T10
本机anchor5参照，不把3090的87.214或5090仅T3的anchor2.5当完整对照。
复用`9_27_imgr10_head_balance_5090.log`中的head_balance_off组；其已记录字段由测试
逐项比对。旧提交未打印的新功能显式关闭，跨提交/环境仍需记录，不能宣称逐位完全一致。
本次仅一个候选，不重新跑基线。data_path/device继续使用本机JSON。

可选独立GPU通路短测（不自动串联、不用于性能）：

```sh
bash scripts/9_28_imgr10_ca_stats_transport_5090.sh --smoke
```

正式运行：

```sh
lrun scripts/9_28_imgr10_ca_stats_transport_5090.sh ./logs/9_28_imgr10_ca_stats_transport_5090.log
```

与3090脚本互不依赖。每任务权重保存已开启，预留至少6GB，文件不含续训统计。
额外开销是Task1–9每任务两次当前训练集的骨干前向和CPU统计变换，没有额外推理开销。
5090近期T3约54.64分钟；共享GPU负载较大，完整T10粗估3～4小时，非承诺结束时间。

## 结果判定

`CATransport`输出scale范围、offset范数、当前训练对齐前后MSE与样本数，证明校正执行。
最终比较同机Average/Last、阶段Old/New、Forgetting和StageAudit错误计数。
若Old与总体成绩没有改善，或只是以New下降换Old，则不替换基线。
它不与间隔损失组合，不预设两者相加会更好。

CPU测试覆盖身份/平移、正定性、范围、只处理旧统计、当前train来源、特征顺序和随机流、
默认关闭的任务生命周期及单候选配方。尚无CUDA短测或性能证据，未启动服务器。
