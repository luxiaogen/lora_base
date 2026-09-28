# 保存每个任务的评估权重

`save_task_weights` 默认关闭；手动运行可加 `--set save_task_weights=true`。
本次 3090 重跑脚本已经开启，无需另外传参数：

```sh
lrun scripts/9_28_imgr10_anchor2p5_save_t10_3090.sh ./logs/9_28_imgr10_anchor2p5_save_t10_3090.log
```

只跑一组 ImageNet-R T10、seed1993、Task0 anchor=2.5。其余训练配置与
`imgr10_anchor2p5_t10_3090.json` 相同；数据路径继续读取本机 JSON。
此前 Average=87.214 是十阶段准确率的平均，不是最终单个模型的准确率；
重跑不承诺逐位复现这个成绩。本次不会额外跑 anchor10 对照。

## 保存内容与位置

每个任务完成 LoRA 合并、CA（Task0 无 CA）和评估后保存一次：

```text
logs/ImageNet_R/10_tasks/ca/<本次唯一prefix>_/checkpoints/<时间戳>/
  task_00.pt
  ...
  task_09.pt
```

每个文件包含完整模型 state_dict（合并后的骨干、分类头和持久 buffer）、
配置、任务编号、类别顺序、各任务类别数、截至该任务的结果。
已合并删除的 LoRA A/B 不另行保存；最终有效更新已在骨干权重中。
保存只复制张量到 CPU，不移动训练模型、不改变随机状态、不额外遍历数据。
不同启动使用独立目录；建议预留至少 6 GB，实际占用取决于模型张量大小。

日志每次打印成功保存的绝对路径，可在服务器查看：

```sh
rg 'Saved task weights:' ./logs/9_28_imgr10_anchor2p5_save_t10_3090.log
```

## 加载范围

`utils.task_weights.load_task_weights(network, path)` 用于加载到按原配置构造的
全新、LoRA 槽为空的同结构网络；函数恢复 state_dict、numtask 并进入 eval 模式。
返回的元数据含 class_order，评估数据必须使用该类别顺序和原预处理。
本次 MANet 的全已见类别推理使用 `network.interface(images)`，不是只计算当前
任务分类头的 `network(images)`；数值后端也应保持 math-SDPA 等原设置。

这是**评估权重**，不是断点续训文件：没有保存优化器、RNG、训练数据加载器状态、
类别原型/协方差或非持久训练掩码，不能据此直接恢复增量训练或 NCM 诊断。

## 已验证与未验证

专项测试覆盖真实 Attention_LoRA 模块的保存/加载、保存不改变参数和随机状态、
两步训练轨迹一致、默认关闭，以及重跑配置只新增保存开关。
本地未执行完整 CUDA ViT 重载评估；服务器本次训练是否成功保存，以每个任务的
`Saved task weights:` 日志和实际 `.pt` 文件为准。脚本不会自动连接或启动服务器。
