# execution_layer

负责 Execution Policy、工具注册、action 校验、预算预留/结算、任务协议、派发与回收、幂等、多轮调度和断点恢复。

子目录：`policy/` 审批和校验，`state/` snapshot/任务恢复，`budget/` 预算事务，`dispatch/` registry 和派发，`workflows/` 多轮业务编排，`slurm/` 批任务。主要入口为 `workflows/run_event_loop.py`、`workflows/run_tool_step.py`、`dispatch/create_tool_registry.py`。

轮次调度中，Agent 从 snapshot 的合法 Branch 摘要中选择本轮批次；
`mc_budget` 定义总上限。所选批次先做三个结构的 Relax 预筛选，随后 Hyperband 负责
MC fidelity、晋级和淘汰。Agent 不能直接指定逐 Branch fidelity。

Linux/Windows 离线审批可设置 `approval_directory`。interactive action 会生成可读 MD、规范 JSON 和独立 decision JSON 后正常退出；文字意见由 Agent 生成新 revision，只有 comment 最后一条非空内容为“同意”或 `approve` 才进入 validation 和执行。详见 `EXECUTION_POLICY.md`。

`create_active_learning_handlers.py` 将 `generate_branches`、`allocate_mc_bohb` 和 `run_calculation_stage` 接入统一入口。BOHB 的 pending/完成记录嵌入 event state；恢复结果先结算共享预算，再由 BOHB 记录观测并决定晋级。

人工上传模式按 MLIP 版本、阶段、批次保存输入。Relax/MC 每批只提交根目录
`GPU.sh`，DFT 在唯一任务目录提交 `GPU.sh`；DFT 输入仍由 atomate 生成。
计算结束后，每轮每个阶段的共享 `results/<任务目录>/` 集中保存所有提交批次的
`result.json`、`task.finished.json` 和最终结构，可直接下载整个 `results/` 到本地同一阶段目录。
本地回收依据原 `task.json` 和校验值识别结果；已有旧批次仍从任务目录回收。
MC 预算中每步成本系数为原估算的 0.1，实际步数与任务分层不变。

`slurm_batch_runner.py` 将已经通过校验且已经预留预算的 `pending_tasks` 组成
Slurm array batch。它只负责文件、提交和回收，不重新选择任务或改写预算。每个任务
目录包含 `task.json`，worker 必须原子写入 `result.json`，随后写入
`task.finished.json`；只有两者都存在才允许回收。下一次调用入口时会自动
扫描终态结果并交给统一预算结算。DFT stage 必须经已有 dispatcher 生成
`generator=atomate` 的 workflow，runner 不生成独立 VASP 输入。

```python
from execution_layer.slurm.slurm_batch_runner import SlurmBatchRunner
from config_layer.defaults.default_slurm_cluster_config import default_slurm_cluster_config

runner = SlurmBatchRunner(
    "outputs/slurm_batches",
    worker_command=["/path/to/py1/python", "/path/to/project_worker.py"],
    dispatcher=dispatcher,
    max_batch_tasks=32,
    max_batch_cost=500.0,
    slurm_options={"partition": "gpu", "time": "24:00:00"},
    stage_profiles=default_slurm_cluster_config(),
    submit=False,  # 检查输入后改为 True
)
```

`worker_command` 会收到 `--manifest <path> --array-index <index>`。可直接使用
`python -m execution_layer.slurm.run_slurm_array_task --executor package.module:function`；它会
读取 manifest 对应的 `input_path`，并原子写入 `result_path`，异常也会形成可回收的
failed 结果。终态至少包含 `status` 和 `actual_cost`。

DFT batch 同样由本 runner 生成 Slurm array，但科学输入仍由 atomate workflow 的
输入写入任务物化；项目不使用 LaunchPad/MongoDB。每个 array 元素进入自己的任务
目录运行 VASP，随后由 `scientific_layer.dft.parse_vasp_result` 原子写出统一
`result.json`。主循环的 state、预算台账和 result 文件是唯一事实来源。

当前站点配置位于 `config_layer/defaults/default_slurm_cluster_config.py`：`deep_search` 和
`relax_and_feature` 共用 CUDA 12.8、32G、V100 GPU profile；DFT profile 保存
VASP 6.5.1/NVHPC 环境和 `mpirun -np ${SLURM_NPROCS} vasp_std`。atomate只负责
DFT输入语义，提交、状态和回收仍由 execution layer 统一管理。

MLIP batch 使用 `scientific_layer.mlip.slurm_executor`。主控侧通过
`create_mlip_task_preparer(manager, phase_references, config)` 把结构路径、固定 branch、
模型版本、MC 增量预算与 checkpoint 固化进 `task.json`；计算节点使用 executor 引用
`scientific_layer.mlip.slurm_executor:execute_mlip_task`。MLIP-MC 和 MLIP-Relax 共用
同一 GPU profile 与 worker 协议。

回收发生在下一次 Agent 决策之前。`apply_scientific_feedback.py` 先识别/复用最终结构的相，
再做 DFT 层状 Fe/Mn 自旋验收、写入 `PhaseDataManager`，并更新分离的 MLIP/DFT 相图、Ehull、覆盖、收益和 Agent 状态摘要，
因此 Agent 不会基于上一版相图做下一步选择。
同一 DFT task 的补磁矩 JSON 经原身份、校验和和不变的最终科学标签验证后，只刷新诊断与科学可用性；
不重复结算预算、追加阶段历史或执行已完成的同帧模型预测。参见 `scientific_layer/dft/TRAINING_RESULTS.md`。
`state_manager.py` 是唯一 Agent 状态边界：在决策前后生成版本化 snapshot，Agent 只读
snapshot，不读取 manager/数据库或可变的完整 workflow state。Action 经统一 schema
规范化后，仍依次经过 Execution Policy、权限、冻结参数和预算校验再派发。

Agent 选出的 Branch 批次使用显式状态机持久化：`selected → relax_pending →
relax_completed → hull_ready → hb_active → completed/failed`。恢复时依据该状态继续，
不再通过 pending task 数量猜测当前阶段。

`step_runner/` 实现无公网计算节点的文件协议：`recover_results`、
`advise_next_actions`、`confirm_action_plan`、`prepare_confirmed_plan`、
`submit_prepared_jobs` 和 `read_runner_status` 各自单一职责。`scheduler_adapter.py` 提供
提交、查询、取消边界；未配置站点命令时返回 `not_configured`，不会猜测或伪造 job id。
plan、batch、task 和 job 分别使用独立标识；重复调用依据 plan id、batch 归属、job id
及 processed task id 保持幂等。
