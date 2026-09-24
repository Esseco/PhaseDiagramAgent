# 计算成本

Proposal 中的下一轮成本以规模模型为初值，并按 stage 使用最近最多 10 个成功任务的
`actual_cost / planned_cost` 校准：30% 初始估计加 70% 历史比例，校准因子限制在
0.25–4.0。审批文件同时展示任务数/MC步数、分阶段成本、已用与预留成本以及执行后
预计剩余预算。

默认值是可校准的相对资源代理，不是实测秒数、核时或费用。预算上限表示用户投入政策，不代表真实后端性能。

基准结构 40 原子：简单检查 0.05，单次 MLIP 弛豫与特征 1，每个“提议并完成 MLIP 弛豫”的 MC 步骤 1，DFT-SP 30，DFT-relax 900。DFT-relax/SP 固定为 30；DFT-SP/单次 MLIP 弛豫在基准规模也是 30。实际比值仍受离子步数、电子收敛、硬件和模型影响。

统一估算入口 `execution_layer/estimate_stage_cost.py`：

```
cost = task_cost × (N / 40)^p × initial_state_count × search_factor × scale
```

简单检查默认 p=1，MLIP 弛豫和 MLIP–MC 默认 p=1.2，DFT-SP 与 DFT-relax 默认 p=3。这里用立方增长表达 DFT 随体系规模快速增长，比数学意义上的指数函数更符合常见平面波 DFT 的经验标度；它仍是可调代理，并非理论复杂度保证。每个 MC step 都包含一次 MLIP 弛豫，因此 `search_factor=计划 MC 步数/1`，10/30/90 fidelity 在40原子基准下分别估算为10/30/90，而不是一次弛豫。未提供原子数时使用40原子并标记 `reference_size_assumed`。

BOHB action 同时保存 `incremental_budget`（本段新增的完整 MC 弛豫步数）和 `planned_relative_cost`（结构规模折算后的相对资源成本）。BOHB 用前者比较 fidelity 与晋级，执行层和总预算使用后者；二者不能互换。

全流程默认总上限为20000。阶段上限可重叠但共享总预算：MLIP–MC 4000，DFT-SP 6000/最多100任务，DFT-relax 15000/最多2任务。40原子的一个 Relax 计900；80原子按立方规模因子计7200，因此两个80原子 Relax 已接近阶段上限。任务数量仍受全局成本和其他阶段开销约束。

DFT action_costs 从 default_budget_rules 派生，正式主动学习以 run_config.budgets 为权威。DFT 检查同时计入运行中的预留和同批次新任务，分别检查全局、阶段成本和任务数。

建议在同一后端、模型和DFT参数下记录原子数、MC步数、离子步数、核数/GPU数、耗时及任务状态，用同组任务资源消耗的中位数校准 task_cost 和指数。CPU核时与GPU时需明确换算权重；不能直接相加。后端 actual_cost 必须换算成同一 relative_cost 单位再提交。优先使用可靠调度器记录中的 GPU 数与实际运行秒数计算 `actual_gpu_core_hours`。没有实测值时 `actual_cost` 保持 null；预算台账另存 `estimated_cost`、`accounted_cost` 和估计依据。已知实际步数可用于释放部分预留，但不能改名为实测 GPU 成本；连实际步数也未知时以预留额作为保守估计上界。

MC 任务分别保存 `max_mc_steps/requested_max_mc_steps`、`patience_steps`、
`min_improvement`、`actual_mc_steps`、`stop_reason`、请求预算、预留成本、实测成本
和估计成本。`patience_steps` 是早停参数，不是实际执行步数。失败、超时和取消若有
已消耗成本同样结算；settlement ID 与 task ID 保证重复回收不重复扣费。

旧状态和已确认配置不自动换算或改写。旧 DFT action_costs=1/4、20/200 与新30/900账本不能直接相加；旧 MC“1000步=4”与新“每个含弛豫的MC步=1”也不兼容。建议新运行采用新默认值，旧运行继续使用原配置完成或明确迁移预算。独立直接调用仍可显式提供旧 action_costs。
