# 2026-09-28：全量anchor短程消融与S学习率候选

## 完整日志补记（替代下方初次快照的未完成状态）

用户补发的同路径3090完整日志SHA256为
`596b7c9bedfb0a3ea063b22a01e97f8b7b22e0cc0a5cd11714f98c6a90c59121`。
四组均完成Task0–2、exit0，未见运行错误；下面初次快照及hash保留作追溯，不再代表当前完成状态。

| 补齐组 | Task0 | Task1 | Task2 | Average | Task2 Old | Task2 New | Forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor5 | 96.37 | 92.43 | 89.98 | 92.9267 | 91.22 | 87.29 | 2.1350 |
| anchor20 | 96.52 | 92.51 | 89.47 | 92.8333 | 90.92 | 86.30 | 2.6300 |

anchor2.5仍是四组Average和Last最高者。与同批anchor5相比Average+0.28pp，
主要来自Task0+0.73pp；Task1相同、Task2+0.11pp（Old+0.15、New相同），
且Forgetting高0.095pp。不能宣称各指标全面改善。
下一步实施[两机确认队列](2026-09-28-anchor2p5-confirmation-plan.md)，尚未启动服务器。

## 日志身份和完成情况

均为ImageNet-R、seed1993、T10类别顺序，仅执行Task0–2；20epochs、CA5、math-SDPA，
全量原训练数据/官方测试，不是holdout成绩。单seed、短程，不是完整T10证据。

| 原日志（均在 /Users/luxiaogen/Desktop/loda_logs/） | SHA256 | 状态 |
|---|---|---|
| 9-28/9_28_imgr10_anchor_fulltrain_t3_3090.log | e67d6d6d53bc3950dde554fff8828e23260780d3c19578d6df9203b48c114631 | a4069b0；anchor0和2.5各完成3任务、exit0；anchor5停在Task2 Epoch17完成/18开始，20未出现 |
| 9-28/9_28_imgr10_slora_lr_t3_5090.log | ce3fdd1a2d1d61fceb7c045b291f6ac0eb01ff309f6894e885fa8327b24255ee | 0ca343b；完成3任务、exit0，约53.48分钟 |
| 9-27/9_27_imgr10_head_balance_3090.log | 4887496ba1de6f27f4078214f39284f5cdc0fe3aa6423d8e8dac89d7abc30060 | 仅复用head_balance_off组前三任务，anchor10 |
| 9-27/9_27_imgr10_head_balance_5090.log | 2d18b056f25586cc850ac25c8d57b46d9b012f0190d3dd6364d4525e6eed52db | 仅复用head_balance_off组前三任务，anchor5 |

本次两份快照未发现traceback/OOM/非有限指标。3090后半不完整不等于报错，
不能由下载快照推断服务器已停止。解析器3/10的CHECK基于total_sessions10，
应结合max_tasks3和exit0判断短程是否完成，不能误报anchor0/2.5未完成。

## 3090 anchor

| Task0独立anchor | Task0 | Task1 Total | Task2 Total | 三阶段Average | Task2 Old | Task2 New | Task2 Forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10（历史本机参考） | 96.52 | 92.28 | 89.72 | 92.8400 | 91.07 | 86.80 | 2.3000 |
| 0 | 96.52 | 92.51 | 89.52 | 92.8500 | 91.14 | 85.97 | 2.2200 |
| 2.5 | 97.10 | 92.43 | 90.09 | 93.2067 | 91.37 | 87.29 | 2.2300 |
| 5（未完成） | 96.37 | 92.43 | — | — | — | — | — |
| 20（未出现） | — | — | — | — | — | — | — |

2.5相对10：Task0 +0.58，Task1 +0.15，Task2 +0.37，Average +0.3667pp；
Task1–2的Total平均提高0.26pp，因此不完全是Task0抬高Average。
Task2 Old/New分别+0.30/+0.49pp，遗忘−0.07pp。
但Task1 Old +0.44、New −0.16pp，不是每阶段同时改善两者。
当前支持2.5作为3090后续工作基线，不支持“全部anchor已比完/最优/完整T10必升”。
按用户要求登记[B1-3090](baselines/B1_3090.md)，保留历史参考不覆盖。

## 5090 S学习率

| 设置 | Task0 | Task1 Total | Task2 Total | 三阶段Average | Task2 Old | Task2 New | Task2 Forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| S=0.02（历史本机参考） | 96.81 | 92.28 | 89.93 | 93.0067 | 91.22 | 87.13 | 2.1350 |
| S=0.01 | 96.81 | 91.82 | 89.67 | 92.7667 | 90.99 | 86.80 | 1.8150 |
| 差值 | 0 | −0.46 | −0.26 | −0.2400 | −0.23 | −0.33 | −0.3200 |

Task1 Old +0.15、New −1.11pp；Task1–2平均Old −0.04、New −0.72pp。
代码确实生效：Task0初始lrs=[0.02,0.02]，Task1/2=[0.02,0.02,0.01]。
遗忘较低但Old绝对准确率也低；不能仅用遗忘降低证明保护更好，历史峰值/学习程度也会影响该指标。
结论：不采纳S学习率减半，5090保持anchor5、S/P学习率0.02，不追加完整T10或系数扫描。

## 可比性与下一步

已运行技能inspect_log/compare_runs并检查原始CNN行。比较器同时看到head_balance_on候选，
该候选明确排除，不能把其head_balance_weight0.1当本次基线配置。
历史基线较早：新脚本多了显式关闭holdout、head_start、蒸馏、history_audit、CA shadow、P-A训练等字段；
旧日志未打印这些字段，不代表已在旧实验启用。已核对当前默认关闭，但跨提交仍非严格同提交配对。
5090有本机JSON修改，已打印有效CLI覆盖；预训练文件/数据hash、运行库版本没有完整匹配证据。
3090只改变anchor且Task0不同是预期效应；5090Task0一致，但不足以证明后续逐位可复现。

下一步如继续训练，只需用3090 anchor2.5完成T10确认，与本机历史anchor10完整T10比较。
本轮未创建/启动该训练，不重跑整个anchor矩阵。不把官方测试选出的2.5当无偏验证结果；
论文需披露选择过程，并最终独立多seed/数据集确认。
