# scientific_layer

包含不依赖 LLM 的科学算法和后端 adapter：`structures/`、`mlip/`、`mc/`、`dft/`、`qbc/`、`bohb/`、`features/`、`surrogate_models/`、`training/`。

输入必须显式给出，输出科学结果或任务描述；本层不决定是否执行 action，不维护正式搜索台账。后端环境沿用既有配置：主控 `py1`，MACE 可由 `py-mace` adapter 调用。

正式流程由 Agent 选择 Branch 批次，Relax 提供最低能量和批次内能量离散度，Hyperband 控制 MLIP–MC fidelity、晋级和淘汰。`bohb/` 中的贝叶斯选择接口保留用于实验，默认不启用。

`mlip/slurm_executor.py` 是 MACE Relax 与 Process_AL_MC 的计算节点适配器：输入是
execution layer 已冻结的自包含任务，输出统一结果。`dft/create_atomate_workflow.py`
使用 atomate 物化 VASP 输入，`dft/parse_vasp_result.py` 解析直接 Slurm 计算结果；
这些模块不管理任务状态或预算。

每个 Process_AL_MC 调用是独立搜索段，保存最大步数、patience、最小改进、实际步数
及停止原因。从结构重新启动不会标为完整 Markov 链 checkpoint 续算。训练产物先作为
候选模型保存；验证只使用初始配置中用户确认的宽松异常阈值，不要求 MAE 单调下降，
且不会自行激活模型。
