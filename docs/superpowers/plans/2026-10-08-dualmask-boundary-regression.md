# DualMask 边界修复与单组基线回归

**Goal:** 修复四类非默认组合问题，保持合法完整基线不变，等3090当前整队结束后重跑ImageNet-R T10 seed1993。

**Spec:** 本文记录用户已确认的“四类问题修复＋等待后重跑一组基线”计划。

**Architecture:** 在现有入口规范配置，只改掩码来源、强度赋值与参数收集；复用原完整基线队列，不新增训练机制。

**Tech Stack:** Python3.9兼容代码、PyTorch、unittest、Bash/Fish、Git。

## Global Constraints

- 修复前参照：8c92912d6a6e44fcb45e7795d72387713ede835e。
- overlap缺省false；有限越界强度截到[0,1]并提示；NaN/Inf在启动前拒绝。
- 单分支与分阶段不兼容；Task0-only不拒绝；单分支末期平均正常支持。
- 合法完整基线、Task0、初始化RNG、优化器和合并方式不变；检查留在入口与测试。
- 原完整DualMask，anchor2.5、20epochs、CA5、math-SDPA；不保存checkpoint，不使用5090。
- 保留所有已有脏文档、output、数据、本机JSON及纯基线调试分支。
- 服务器当前队列运行时不更新代码；只等真实外层PID，退出后检查完整成功记录。

## Review Focus

- CLI覆盖后的配置才规范，日志/指纹必须记录实际值。
- 双开置换只置换一次，单开与默认不变。
- 合成B只平均一次，A/旧头/骨干不平均。
- 选择一组时不能预检或短测其他数据集；默认仍12组。
- lwait退出不代表原队列成功，失败或不完整不能接跑。

### Task 1: 四类边界修复

- [x] 补回归测试并观察RED；仅正常旧行为测试允许先通过。
- [x] 入口normalize_dualmask_config在apply_overrides后执行；规范overlap，截断固定保护强度，拒绝增量单分支分阶段。
- [x] P置换从reference_protect生成；before_task固定强度赋值截断。
- [x] 原型准备overlap缺省false；末期平均收集实际存在的当前任务B及当前头，保留顺序。
- [x] 与参照提交逐值比较Task0/1前向、loss、梯度、SGD一步与合并；检查merge-once。
- [x] 专项测试：env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest test.test_dualmask_boundary_fixes test.test_protect_position test.test_p_permission_release test.test_compact_dualmask test.test_sp_staged test.test_late_weight_average。

### Task 2: 基线队列单组选择

- [x] 补选择一组、跨数据集选择、默认12组、仅选中数据预检、resume子集一致性测试并观察RED。
- [x] run_baseline_suite增加--modes；默认不变，按选中数据集去重短测，按选中顺序正式训练，新启动不resume旧目录。
- [x] Bash入口遵守已选择PYTHON，服务器使用原conda解释器。
- [x] 专项测试：env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest test.test_baseline_suite。
- [x] 全量：env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s test。

### Task 3: 审查、推送和lwait接续

- [ ] 独立代码审查；必要修复RED→GREEN；提交只包含本轮文件并推送，交付完整SHA。
- [ ] 用active.json、ps及GPU核实3090当前外层PID、原环境、配置哈希；等待时不更新项目。
- [ ] 用现有Fish lwait等待整个队列，检查原计划所有正式组完成退出0后再安全快进到修复SHA。
- [ ] GitHub服务器认证不可用时，将已推送提交的增量bundle预置到明确临时路径，结束后才fetch/ff；不覆盖本机JSON。
- [ ] 只启动--modes imgr10_seed1993；独立日志/新目录；一轮两任务短测成功后完整T10。
- [ ] 交付等待PID/日志；真正启动后返回队列/训练PID和源码/配置指纹，完整训练后报告结果。

审查补充：已复现队列快照仍记录越界原值。将规范逻辑移到
`scripts/dualmask_config.py` 供入口与快照共用；快照保留原命令但记录实际值，
不重复提示。相关新测试先失败再通过；当前全量757项，742通过、15跳过。
