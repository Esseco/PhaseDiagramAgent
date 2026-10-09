# run

当前聊天入口为 Studio：使用 langgraph dev（环境设置见 docs/STUDIO_LOCAL.md），或运行 python -m run.studio_service --runtime-config "E:\0-FM-PhaseDiagram\agent_runtime.json"。
可选项目选择器：python -m run.local_project_launcher。不再提供重复 cmd 入口。
依赖统一为根目录 requirements.txt；Studio API 端口2024，控制/审批端口8765。
详情见 [正式启动说明](../docs/STUDIO_LOCAL.md)。

配置确认后，如需修改持久参数，可在 Agent 对话中明确说“把配置文件的 `system.H_generation.first_round_max_det_H` 改为 12 并写入”。程序只写可编辑的 `search_config.project.json`，校验源文件未被其他编辑改动；然后发送“读取配置 JSON”审核并确认新快照。旧快照与正在运行任务不会被覆盖。`size_max` 控制 H 枚举范围，`first_round_max_det_H` 控制首轮生成动作；两者不是同一个值。单轮限制也可说“H 首轮上限设为 12，重新生成 branch”，此值只作用于该次生成，不改持久配置。审批时请核对实际动作参数 `max_det_H`；结果会分别显示本轮设定上限与入选 H 的实际范围。

工作区根目录与 Agent 模型可同一行输入，但会分别解析；例如 `E:\0-FM-PhaseDiagram agent：V4.1flash` 不会成为一个目录名。若已确认配置中的根路径与期望路径不同，搜索对话不会把“输出目录”误提交为科学 action，也不会擅自迁移台账；应在正确目录建立新项目，或备份并核对旧运行数据后进行迁移，再确认配置版本。启动时发现路径混有模型文字会明确报错。

人工意见导致 Agent 修订科学 action 时，程序会校验工具名，并为缺少内部编号的正式动作生成稳定 `task_key`；无效动作不会显示为可批准建议。输出目录请求只核对已确认配置，不触发去重、Relax 等科学任务。


## 超算分步入口

暂存版到根目录的功能归并记录见 [SUPERCOMPUTER_MIGRATION.md](SUPERCOMPUTER_MIGRATION.md)。
根目录是唯一正式源码和主状态所有者；远端目录只是不可变部署/结果载体，不维护第二入口。

外部部署统一使用：

```bash
python -m run.step_runner --config confirmed_session.json --node-role compute recover
python -m run.step_runner --config confirmed_session.json --node-role login advise --config-version CONFIG_VERSION
python -m run.step_runner --node-role login confirm --plan PLAN.json --approve
python -m run.step_runner --config confirmed_session.json --node-role compute prepare --plan PLAN.json --config-version CONFIG_VERSION
python -m run.step_runner --config confirmed_session.json --node-role login submit
python -m run.step_runner --config confirmed_session.json status
```

全局参数应写在子命令之前。`advise` 是唯一调用联网 LLM 的步骤；`recover/prepare` 不导入
或调用 LLM。`prepare` 只接受 `status=confirmed` 且配置版本一致的 plan。正式科学 action
通过 `supercomputer.worker.action_executor=package.module:function` 接回现有工具注册表和
工作流；没有配置该适配器且计划要求新建任务时会明确停止。

`supercomputer.scheduler.submit_command/query_command/cancel_command` 默认均为空。管理员
提供命令后再写入已确认配置；为空时 `submit` 返回 `not_configured`，不会假装提交成功。
状态、摘要、计划和任务目录路径统一由 `supercomputer.paths` 配置。

`--config` 只接受带完整 `confirmed_snapshot` 的已确认配置会话；普通 JSON 不会直接成为
有效运行配置。提交处于 `submitting/submission_uncertain` 时必须先按 job id 或 submission
token 查询，无法确认则维持暂停，不会再次提交。

每个新体系先完成小规模初始 MLIP 体检，至少包含端点与中间允许相。微调模型在独立验证
通过但收益不可辨认时保留为候选；默认要求模型激活单独审批。激活后只刷新关键结构，
明显失败、严重超限或近 hull 排序反转会暂停并建议人工回退。模型、DFT 数据版本、验证
摘要和审批理由保留在状态中，旧能量池只标记 stale，不删除或混入新版本。

`run_workflow` 可通过 `initial_long_term_advice=["优先 DFT-SP"]` 初始化长期建议；恢复时沿用 state 中的修订版本。离线审批的 decision JSON 新增 `long_term_advice` 字段，完整列表替换、null 不修改、[] 清空。详见 [决策记忆](../data_layer/DECISION_MEMORY.md)。

唯一公开入口是 `run_workflow`。它默认注册 Branch 生成、BOHB 分配、QBC→DFT
候选选择和统一计算阶段 handler；`run_active_learning_cycle.py` 仅保留为旧调用兼容
模块。BOHB/DFT 子任务与 event state 共用任务、预算预留和恢复记录。QBC 只计算
不确定性，实际 action 仍由 Agent 提出并经过唯一 Execution Policy。

`run` 只负责跨层组合。唯一正式入口是 `run.main.run_workflow()`，并由包顶层导出为 `run_workflow`。

```python
from run import default_run_config, run_workflow

result = run_workflow(
    manager,
    phase_references,
    default_run_config(),
    confirmed_config_session,
    dispatcher=dispatcher,
    agent_client=agent_client,
    execution_mode="interactive",
    invocation_id="cycle-0001",
    max_steps=20,
)
```

`run_workflow` 只接受已确认配置，并由 `build_effective_run_config()` 生成唯一有效配置。第一次 interactive 调用返回 `awaiting_approval`，使用相同 `invocation_id`、保存的 `state` 和 `human_feedback=approve/modify/reject` 恢复。autonomous 模式在 `max_steps` 范围内持续决策，遇到 pending/running、暂停、预算耗尽、失败或收敛即返回。

传入 `task_runner=SlurmBatchRunner(...)` 后仍使用同一个 `run_workflow`：入口先扫描
上一批 `result.json`，完成回收和预算结算；Agent/BOHB 产生新任务后，入口再按已经
预留的预算生成一个 Slurm batch。`submit=False` 只生成文件，`submit=True` 才调用
`sbatch`。DFT 仅允许 dispatcher 的 atomate workflow。

内部实现：

- `main.py`：组装 effective config、Tool Registry 和可恢复事件循环。
- `run_active_learning_cycle.py`：保留的内部批处理实现，用于已有测试与细粒度组合。
- `run_pipeline.py`、`search_iteration.py`：确定性搜索迭代。
- `resume_pipeline.py`：从台账和状态文件恢复。
- `run_confirmed_active_learning_cycle.py`：配置版本绑定。
- `default_run_config.py`：运行时默认值；科学后端仍通过 adapter/dispatcher 注入。

这些模块用于测试和内部组合，不是额外的正式启动入口。正式入口中的每个 action 均经过 Agent proposal → Execution Policy → validation → Tool Registry → handler；机器相关计算通过 `dispatcher` 或 runtime adapter 注入，`handlers` 仅用于替换或扩展默认行为。
Studio命令行启动与审批说明见 [STUDIO_LOCAL.md](../docs/STUDIO_LOCAL.md)。
