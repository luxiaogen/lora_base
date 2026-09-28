# 3090 anchor2.5：离线特征诊断

只读取已保存的最终模型和现有 ImageNet-R train/test 目录，不训练、不修改数据、不启动 W&B。
读取 checkpoint 中的模型配置和类别顺序；本机 JSON 只提供 data_path/device。
默认权重来自原日志 `9_28_imgr10_anchor2p5_save_t10_3090.log`：
Average87.281，最终模型 Last82.67。单个最终 checkpoint 只能复测 Last，不能复测十任务 Average。

## 安装和运行

在服务器现有 Python3.9 训练环境、原项目目录下，一次性补齐绘图依赖：

```bash
python -m pip install -r requirements-feature-geometry.txt
```

然后按现有 lrun 习惯执行：

```bash
lrun scripts/9_28_imgr10_feature_geometry_3090.sh ./logs/9_28_imgr10_feature_geometry_3090.log
```

脚本默认读取确切权重，不搜索最新文件：

```text
logs/ImageNet_R/10_tasks/ca/imgr10_anchor2p5_save_t10_3090_anchor2p5_20260928_133705_194333_/checkpoints/20260928_133709_314001/task_09.pt
```

若文件移动了，可用 Bash 传入实际文件与新输出目录：

```bash
bash scripts/9_28_imgr10_feature_geometry_3090.sh /实际路径/task_09.pt logs/feature_geometry/my_baseline
```

启动先加载绘图库，缺包不会等特征提取结束才暴露。模型构造仍使用项目原有 AugReg ViT
加载路径，因此服务器需保留原预训练文件/缓存。脚本不使用 DataManager.download_data 的自动划分逻辑；
缺 train/test 目录会由只读 ImageFolder 报错，不会移动或删除图片。

## 输出

默认目录 `logs/feature_geometry/anchor2p5_3090_<时间戳>/`，启动时打印：

- `features.npz`：全部已见训练/测试图片的最终 CLS 特征、标签、索引、全部已见分类头。
- `sample_paths.json`：train/test 索引对应的图片路径（分享前留意本地路径信息）。
- `metadata.json`：checkpoint SHA256、类别顺序、模型配置、代码提交/dirty状态、环境、数据路径。
- `figures/01_confusion.png/pdf`：全部类别混淆矩阵，真实类别行、预测类别列、行百分比。
- `figures/02_tsne_pairs.png/pdf`：按双向错误计数排序的前三对，每类固定随机抽至多100张。
  每对同时展示两个 perplexity；样本少时按样本数下调。不同面板不能比较绝对坐标/距离。
- `figures/03_center_head_alignment.png/pdf`：训练中心之间的余弦相似度；中心对自身/最强其他分类头的相似度。
- `classes.csv`：全部类别的样本数、准确率、类内散布、最近异类中心距离、分离/散布比、分类头对齐和 margin。
- `samples.csv`、`confusion_counts.csv`、`selected_pairs.csv`、`tsne_coordinates.csv`、`geometry.npz`：可追溯数值。
- `summary.json`：重载 Last 与 checkpoint Last 的差异、错误计数；不是自动认定性能有效。

CSV/JSON/图在 figures 目录下。优先检查重载准确率应接近82.67（保存值只有两位小数）；
若明显不一致，先核对数据、checkpoint和数值后端，不继续解释几何。

缓存完成后，只重画图，不使用 GPU、不加载模型：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/plot_feature_geometry.py --cache logs/feature_geometry/具体时间戳目录
```

上面是 Bash 命令；Fish 可用 `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 ...`。
也可把整个输出目录复制回本地再画图。提取耗时取决于 GPU/磁盘，日志逐25批报告进度；
没有20轮训练，但会对全部已见 train/test 图片各前向一次。当前没有实机计时，不承诺固定分钟数。

## 解释边界

类中心 = 每张特征 L2归一化后求均值，再归一化；分类头也归一化。
散布 = 样本到本类中心的均方根弦距离；分离 = 最近异类中心弦距离。
margin = 正确分类头余弦分数 − 最强错误分类头余弦分数。
这些统计均在原始768维计算，t-SNE只用于展示。

历史训练图片只用于离线诊断，不能把重提取的旧类中心用于新方法训练或正式 CIL 推理。
最近中心准确率是这种离线参考，不是合法的无旧数据方法成绩。
按测试错误选择类别对仅用于描述现象，不能用它挑损失权重/最佳 epoch 或作为独立验证。
同一 checkpoint 只有 CA 后状态，不能还原 CA 前预测；CA自身不改变骨干特征。

sciplot 技能安装缺少所引用的 scripts/sciplot.py 与参考协议，使用自包含 Matplotlib 回退。
`visualspec.json` 声明映射/限制，状态为 render_only；合成样例已做版式检查，不是实验结果，
服务器真实输出仍需视觉检查。本地无服务器权重/数据，不能声称已完成真实 CUDA 提取。

## 本地验证（2026-09-28）

13项专项测试通过（9项特征诊断 +4项权重保存/加载），包含公式、类别重映射、
只读图片加载、缺少划分时不修改目录、推理不改权重、三图PNG/PDF生成。
Python编译、Python3.9语法检查、Bash语法和CLI帮助检查通过。
全量发现312项：310项通过，2个旧测试模块因本机缺少easydict未能加载；独立审查通过。
本地合成图用于版式检查，真实 checkpoint/图片未在本机执行。
服务器仍需完成真正的模型加载与重载准确率核对；本地测试不代表数据/权重一定存在。
