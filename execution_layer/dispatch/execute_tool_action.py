"""Execute a validated tool once; persisted receipts guard side effects."""
from execution_layer.policy.file_approval import proposal_hash


def execute_tool_action(action: dict, *, registry: dict, context: dict) -> dict:
    item = registry.get(action.get("tool")) or {}
    handler = item.get("handler")
    tool = action.get("tool")
    if handler is None:
        return {"status": "not_configured", "tool": tool, "result": None,
                "error": "tool handler not configured"}
    state_path = context.get("state_path") or (context.get("effective_config") or {}).get("state_path")
    identity = None
    if state_path and tool not in {"check_convergence", "pause_search"}:
        from execution_layer.state.execution_receipts import begin_execution
        identity = {"invocation_id": context.get("invocation_id") or context.get("approval_record_id") or action.get("task_key"),
                    "config_version": context.get("config_version"),
                    "action_hash": proposal_hash(action), "tool": tool}
        receipt = begin_execution(state_path, identity)
        if not receipt["allowed"]:
            return {"status": "execution_reconciliation_required", "tool": tool, "result": None,
                    "error": "执行收据已存在，需核对已有文件/任务和结果入账；未重复执行。",
                    "receipt": receipt}
    try:
        result = handler(action=action, context=context)
        if tool == "check_convergence":
            from analysis_layer.convergence.normalize_convergence_result import normalize_convergence_result
            result = normalize_convergence_result(result)
        execution = {"status": "completed", "tool": tool, "result": result, "error": None}
    except Exception as error:
        execution = {"status": "failed", "tool": tool, "result": None,
                     "error": f"{type(error).__name__}: {error}"}
    if identity is not None:
        from execution_layer.state.execution_receipts import record_execution_return
        record_execution_return(state_path, identity, execution["status"])
    return execution
