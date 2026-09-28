# 3090 anchor消融：全量训练、官方测试、仅Task0–2

顺序：anchor0、2.5、5、20，四次均运行，不筛选、不自动续跑T10。
anchor10复用 `9_27_imgr10_head_balance_3090.log` 的 head_balance_off 全量基线前三任务。
该历史参考不是本提交同步重跑；stage_audit同样开启，只读记录，不改变训练目标。
不复用此前80%训练数据的holdout成绩。完整T10日志的前三任务可作T3参考，
但本次结果只代表T10类别顺序下的前三任务，不是完整T10结果。

固定：ImageNet-R、seed1993、init_cls/increment20、total_sessions10、max_tasks3，
Task0/后续20epochs、CA5、math-SDPA、冻结A、双门、掩码正则0.01、原随机分类头。
仅切换 `dual_mask_anchor_reg_weight`，独立anchor仍只用于Task0。
显式关闭 `incremental_holdout` 和 `task0_validation_enabled`，全部原训练数据进入训练；
官方测试集仅用于报告预先固定的四组消融，不用于早停/续跑。
保留方法原有的W0 competence训练集统计，不改变其算法。
本机JSON data_path不覆盖，不修改训练代码或默认配置。

```bash
# 可选一轮通路检查，不是正式成绩
bash scripts/9_28_imgr10_anchor_fulltrain_t3_3090.sh --smoke
# 正式四组（没有自动短测/测试门禁）
lrun scripts/9_28_imgr10_anchor_fulltrain_t3_3090.sh ./logs/9_28_imgr10_anchor_fulltrain_t3_3090.log
```

按近期3090全量T3每组约22–25分钟估计，共约1.5–2小时，视服务器占用浮动。
每组独立日志保存在 `logs/shell_logs/imgr10_anchor_fulltrain_t3_3090/<timestamp>/`，
包含实际命令和退出码；外层日志记录commit和工作区状态。失败组不阻断其他组，最终返回失败状态。
全部报告Task0、三阶段Average、Task2 Last/Old/New/Forgetting，不只列最好一组。
GPU尚未在本地运行，脚本测试不证明性能或耗时。
