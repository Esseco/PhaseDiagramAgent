# Execution Policy Layer

每版 proposal 包含下一轮计算量、分阶段校准成本、总成本和预计剩余预算。DFT 没有
固定配额：Agent 可以建议 0 个或少量代表性任务，其职责是校准/纠正 MLIP，而不是
覆盖大部分候选。

Execution Policy 位于 Agent proposal 与现有 action validation 之间，只控制 action 是否继续，不进行科学计算、合法性校验或预算扣账。

## Interactive

第一次调用只生成并保存 proposal：

```python
proposed = run_tool_step(
    state,
    session,
    registry=registry,
    agent_client=agent_client,
    execution_mode="interactive",
    invocation_id="round-12-action-1",
)
```

返回 `status="awaiting_approval"`，其中 `agent_proposal` 包含状态分析、action、参数、原因、预计成本和预期目的。

使用相同 `invocation_id` 恢复并处理反馈。只有 comment 最后一条非空内容精确为 `approve` 或“同意”才会执行：

```python
approved = run_tool_step(
    proposed["state"],
    session,
    registry=registry,
    execution_mode="interactive",
    human_feedback={"decision": "approve", "comment": "同意"},
    invocation_id="round-12-action-1",
)
```

文字意见不会执行 action，而是交给 Agent 分析并生成下一版 proposal：

```python
human_feedback={
    "decision": "comment",
    "comment": "减少本轮预算，并说明为什么选择这个区域",
}
```

修订可以重复多轮。每一版都保持 `awaiting_approval`，直到明确批准。也可以人工修改后批准：

```python
human_feedback={
    "decision": "modify",
    "comment": "降低本轮预算\napprove",
    "modifications": {"budget": 0.5, "parameters": {"steps": 100}},
}
```

也可传 `{"decision": "reject"}`。修改后的 action 会重新经过完整 validation；interactive 模式不会把非法修改静默替换为其他 action。

## 文件审批

`run_workflow(..., approval_directory="run_state/approvals")` 会按 action 和 revision 写出 `proposal-rNNN.json`、`proposal-rNNN.md` 与可填写的 `decision-rNNN.json`。程序随后正常返回，不占用终端或计算节点。编辑 decision 文件后，用相同 state、`invocation_id` 和 approval directory 重新启动：普通意见生成下一版；最终 comment 为“同意”或 `approve` 时才继续执行。proposal hash、revision 和 ID 必须匹配，防止批准过期提案。

## Autonomous、dry-run 和 replay

- `autonomous`：Policy 自动批准，然后 validation 和执行。
- `dry_run`：Policy 自动批准并执行 validation，但不调用 handler。
- `replay`：读取旧记录的 `final_action`，重新 validation 后执行。

```python
replayed = run_tool_step(
    state,
    session,
    registry=registry,
    execution_mode="replay",
    replay_record=old_record,
)
```

每次 action 在 `state["action_records"]` 保存：

```text
agent_proposal
human_feedback
final_action
execution_result
```

所有获准执行的 action 仍由原 `validate_tool_action()` 检查 config version、冻结参数、预算和 tool permission。

QBC/DFT acquisition 也使用同一 Execution Policy 语义。它在 policy 放行后使用
原有 `validate_dft_agent_decisions()` 完成候选、action 白名单、配置版本、去重和
DFT 预算校验，再进行预算预留与提交；`dry_run` 只校验，不预留也不提交。
