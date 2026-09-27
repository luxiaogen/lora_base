# 新分类头双向训练：ImageNet-R seed1993

实验卡：`docs/experiments/实验日志/DualMask/分类头训练/DM-B-260927-001.md`。
这是待验证候选，不是已证实的性能改进。原项目、原实验分支；不启动远程机器。

## 改动

仅新增 `head_balance_weight`，默认0。本次候选固定0.1，Task0完全跳过。
Task1起，在原分类/正则损失之外加全局余弦CE，scale沿用20：

- 当前新图片：复用本步特征并detach，不额外前向ViT；标签转换为全局类编号。
- 旧类：复用已有均值/协方差，沿用CA年龄缩放均值；每个任务开始时生成256条/旧类的临时CPU伪特征池。
- 每步从每个旧类抽1条；旧分数由固定旧头计算，新分数由当前头计算，所有已见类别共同竞争。
- 新样本CE先对本批每个出现的新类取均值，再对这些类平均；旧样本每类1条，直接平均。
  两部分按新类数/已见类数、旧类数/已见类数加权，避免伪特征和新batch的数量决定损失权重。
- 额外损失仅更新新头；旧头权重、旧伪特征、新图片特征都detach。原分类损失仍更新LoRA和新头。
- 原Gaussian CA（全部已见头可训练）、双门、merge、统计更新、推理保持原样。

总损失：原分类损失 + 原正则 + head_balance_weight × 全局平衡CE。
不加旧类竞争hinge，不改CA真实特征选项。不使用旧图、测试标签调参、任务ID推理或额外推理前向。

边界：只保证额外损失的直接梯度隔离；新头改变后，后续原损失可能间接改变LoRA轨迹。
历史统计可能过时，本方案没有解决该近似误差。最终CA也可能抵消训练期收益，需要完整T10检验。
伪特征使用独立CPU generator，不消耗训练/CA的全局随机流；不承诺不同GPU逐位一致。

## 两机安排

| 机器 | 本机参照 | 本机候选 | anchor |
| --- | --- | --- | ---: |
| 3090 | head_balance_weight=0 | head_balance_weight=0.1 | 两组均10 |
| 5090 | head_balance_weight=0 | head_balance_weight=0.1 | 两组均5 |

各2次完整T10，Task0/增量20轮、CA5、math-SDPA，seed1993。其余沿用B0。
3090是B0加本轮明确的anchor10覆盖，不修改冻结的B0文件。不能用昨日anchor5成绩充当3090参照；
两机anchor不同，不把机器间差值当算法收益。数据路径仍取本机`exps/dlora/imgr10.json`。

## 执行（在项目根目录，已激活训练环境）

先按需单独运行短测，成功与否由日志人工判断；正式队列不会自动启动短测、测试或性能断言。

```bash
# 3090 短测：两组Task0–1，每阶段1轮；只检查代码路径，不评价准确率
bash scripts/9_27_imgr10_head_balance_3090.sh --smoke

# 3090 正式：先anchor10基线，再anchor10候选
lrun scripts/9_27_imgr10_head_balance_3090.sh ./logs/9_27_imgr10_head_balance_3090.log

# 5090 短测
bash scripts/9_27_imgr10_head_balance_5090.sh --smoke

# 5090 正式：先anchor5基线，再anchor5候选
lrun scripts/9_27_imgr10_head_balance_5090.sh ./logs/9_27_imgr10_head_balance_5090.log
```

`bash script.sh --dry-run`只打印命令。每次运行prefix含机器、时间和full/smoke，日志不混用。
额外CLI `--set`参数放末尾覆盖；正式配对不要单独改一组的公共设置。
队列失败会记录FAIL并继续下一组，最终返回非零；没有自动重试。

预计：3090两组约3小时；5090约3～6小时，最近同配方单组88～178分钟波动很大。
这是旧日志外推，新候选增加采样和分类头计算，CUDA实际开销尚未测量，不保证结束时间。

## 观察什么

- Task1开始出现 `HeadBalance`，Task0无此日志。候选每epoch记录：
  `head_balance_ce/new_ce/old_ce/weighted/new_to_old/old_to_new`。
  这些是当前训练batch/旧伪特征上的诊断，不是旧测试集准确率。
- 沿用StageAudit，查看pre_merge、post_merge、post_ca的双向错误和Old/New；最终比较Average/Last/Forgetting。
- 保留条件：新→旧改善不以旧→新明显增加为代价，Old基本保持，Average/Last有净收益；
  若只是New换Old，不替换基线。单seed小差异不宣称统计显著，不按测试最高点扫权重。

无额外可训练参数、无额外持久统计；最大临时特征池约135MiB CPU（180×256×768×4字节），
构建时列表与stack可使该部分瞬时约270MiB。GPU顺序处理各类协方差；训练结束释放CPU池。
这是复用已有CA统计，不是“完全无存储”，也不能声称训练时间不变。

独立测试：`python -m unittest test.test_head_balance`。不会由上述脚本自动调用。
