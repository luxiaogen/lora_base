# CA边界加权：单条基线轨迹上的分类头分叉

3090，ImageNet-R T10 seed1993，anchor10、20 epochs、CA5、math-SDPA。
每个Task1–9合并并提取统计后，从同一分类头起点先跑候选CA并报告，
恢复分类头与Torch CPU/CUDA RNG，再运行原CA。后续任务仅沿原CA基线继续。
候选不是一条完整累积T10轨迹；不能把各阶段候选拼成正式方法的Average/Forgetting。

每轮原有每类256条Gaussian特征保持不变。候选根据该轮开始时的原始分类头logits，
计算gap=正确分数−最高错误分数，raw_weight=1+exp(-abs(gap)/0.1)。
每类256条权重除以本类均值，每类总权重不变，最终权重在0.5至2之间。
靠近边界（包括边界两侧）的样本适度强调；不是单调放大所有误分极端尾部。
权重detach且该轮固定；CA损失归一化、优化器、步数、样本数均不变。
候选训练及权重计算只增加分类头计算；每个增量任务额外评估一次候选，会遍历测试集进行骨干前向。
只支持本轮Gaussian配置，不与real_new混用。

日志：CABoundary证明权重实际变化；StageAudit中boundary_shadow_post_ca为候选，
post_ca为恢复后基线。比较同任务Total/Old/New及错误类型。不使用测试报告挑权重。
当前没有独立旧类holdout，此轮是诊断，不是独立验证集筛选。
启用候选不会更新公共默认配置。没有自动测试/assert/短测阻断训练。

运行：lrun scripts/9_27_imgr10_ca_boundary_3090.sh ./logs/9_27_imgr10_ca_boundary_3090.log
可选短测：bash scripts/9_27_imgr10_ca_boundary_3090.sh --smoke
脚本只读本机JSON数据路径。CPU测试不代表CUDA训练或性能提升已验证。
