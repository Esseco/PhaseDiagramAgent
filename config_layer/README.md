# config_layer

成本以 `defaults/default_budget_rules.py` 为默认来源，DFT 决策成本从其派生。规模模型、SP/Relax 比例与校准说明见 [COST_MODEL.md](COST_MODEL.md)。

负责体系定义、搜索边界、冻结参数、预算与收敛规则，以及配置草稿、对话确认和不可变版本快照。

子目录：`defaults/` 默认配置，`schema/` 配置与 Action/State 合同，`session/` 草稿、确认与版本，`runtime/` effective config 和预算扩展。主要入口分别为 `defaults/default_layered_search_config.py`、`session/create_config_draft.py`、`session/confirm_config_snapshot.py`、`schema/validate_search_config.py`、`runtime/build_effective_run_config.py`。

`defaults/default_slurm_cluster_config.py` 保存站点 Slurm profile：MLIP-MC 与批量 MLIP-Relax
共用 GPU 环境；VASP/NVHPC profile 供文件式 atomate DFT batch 使用。这些是
机器配置，不改变已确认的科学参数或预算。
`schema/action_state_schema.py` 定义统一 Action/StateSnapshot 合同；
`schema/state_schema_migrations.py` 执行旧 snapshot 的前向迁移，并拒绝无法理解的未来版本。

配置对话由 `session/create_config_draft.py` 首先询问是否采用默认参数；
`session/answer_default_parameter_prompt.py` 在用户选择默认后输出完整参数合集，并接受
JSON path 形式的选择性覆盖。`defaults/default_mace_committee_config.py` 固定四组已审核
的 committee 参数，成员 0 始终是 Relax/MC 主模型；其余成员只参与最终结构预测和 QBC。
启用微调前必须在配置草稿中填写并确认 `mlip_finetune.validation` 的能量 MAE、关键失败
比例和近 hull 排序反转上限，以及 `refresh_validation` 的激活后复核阈值。缺失时模型
保持候选并暂停确认，Agent 不会临时生成阈值。训练完成不等于激活；激活和回退均保留
模型、训练/验证数据版本及用户理由。
