# run

## Open WebUI 对话入口

项目提供 OpenAI-compatible 本地聊天服务。默认读取项目内本地运行时配置：

```bash
python -m run.open_webui_api
```

复制 `open_webui_runtime.example.json` 为 `open_webui_runtime.json` 后填写本地路径；
仍可用 `--handler-factory package.module:function` 覆盖。入口把 Open WebUI 的真实 user 消息接入现有 `run_workflow`，每次最多推进
一个 action；普通建议可在核对后直接回复“同意”批准或“拒绝”，敏感操作仍需在本机认证审批页确认，并仍经过项目
Execution Policy。启动参数、运行时工厂和
Open WebUI 配置见 [Open WebUI 集成说明](OPEN_WEBUI.md)。

宿主机审批页为 `http://127.0.0.1:8765/phase/approval`。输入独立 control token
后可核对 proposal hash、状态/配置/模型版本、目标、参数、成本与项目生成图表；
过期页面会被拒绝。注册动作 `prepare_local_batch_files` 只在运行配置的
`local_action_directory` 下生成 dry-run 清单和校验值，不提交任务，也不覆盖
不一致的已有目录。

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
