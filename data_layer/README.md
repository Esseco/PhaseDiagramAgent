# data_layer

`memory/decision_memory.py` 保存人工长期建议及修改版本，随现有 state 恢复。初始化与文件修订示例见 [DECISION_MEMORY.md](DECISION_MEMORY.md)。普通审批评论不会自动升级为长期建议。

决策记忆明确分为 `long_term` 与 `short_term`：长期层只接受人工显式写入的体系知识、
物理先验、冻结参数建议和搜索规则；短期层由统一 State snapshot 自动替换，记录近期
action、收益、失败任务和当前搜索状态，绝不自动晋升为长期知识。

负责稳定编号、结构与结果台账、任务历史、模型/委员会版本记录、保存加载和恢复状态。历史 JSON 字段与稳定 ID 算法保持兼容。

子目录：`ledger/` 编号和科学/任务台账，`memory/` 决策记忆，`state/` 调度状态保存恢复，`models/` 模型版本与委员会清单。`ledger/branch_energy_pool_ledger.py` 以 `system_id + mlip_version + energy_basis` 隔离能量池。本层保存事实，不做策略决策或凸包分析。
