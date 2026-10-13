"""Execute a validated tool once; persisted receipts guard side effects."""

from copy import deepcopy

from phase_agent.tools.policy.file_approval import proposal_hash


def execute_tool_action(action: dict, *, registry: dict, context: dict) -> dict:
    execution = _execute_tool_action(action, registry=registry, context=context)
    if "operation_outcome" not in execution:
        from phase_agent.tools.dispatch.operation_outcome import operation_outcome

        execution["operation_outcome"] = operation_outcome(action, execution, context)
    return execution


def _execute_tool_action(action: dict, *, registry: dict, context: dict) -> dict:
    from phase_agent.graphs.cancellation import check_cancelled

    check_cancelled()
    from phase_agent.configuration.schema.action_state_schema import normalize_action
    from phase_agent.tools.dispatch.scientific_operations import operation_binding_errors

    try:
        action = normalize_action(action) if action.get("operation") else deepcopy(action)
    except (ValueError, TypeError) as error:
        return {
            "status": "rejected",
            "tool": action.get("tool"),
            "result": None,
            "error": str(error),
        }
    state = context.get("event_state") or {}
    model = state.get("active_model") or (context.get("effective_config") or {}).get("mlip") or {}
    errors = operation_binding_errors(
        action,
        config_version=context.get("config_version") or state.get("confirmed_config_version"),
        model_version=state.get("active_model_version")
        or model.get("version")
        or model.get("name"),
    )
    if errors:
        return {
            "status": "rejected",
            "tool": action.get("tool"),
            "result": None,
            "error": ";".join(errors),
        }
    from phase_agent.tools.workflows.reference_dft_startup import blocked_action

    blocked = blocked_action(context.get("event_state") or {}, action.get("tool"))
    if blocked:
        return {"status": "blocked", "tool": action.get("tool"), "result": None, "error": blocked}
    item = registry.get(action.get("tool")) or {}
    handler = item.get("handler")
    tool = action.get("tool")
    if handler is None:
        return {
            "status": "not_configured",
            "tool": tool,
            "result": None,
            "error": "tool handler not configured",
        }
    state_path = context.get("state_path") or (context.get("effective_config") or {}).get(
        "state_path"
    )
    identity = None
    if state_path and tool not in {"check_convergence", "pause_search"}:
        from phase_agent.tools.state.execution_receipts import begin_execution

        identity = {
            "invocation_id": context.get("invocation_id")
            or context.get("approval_record_id")
            or action.get("task_key"),
            "config_version": context.get("config_version"),
            "action_hash": proposal_hash(action),
            "tool": tool,
        }
        receipt = begin_execution(state_path, identity, action=action)
        if not receipt["allowed"]:
            return {
                "status": "execution_reconciliation_required",
                "tool": tool,
                "result": None,
                "error": "执行收据已存在，需核对已有文件/任务和结果入账；未重复执行。",
                "receipt": receipt,
            }
    try:
        # Handlers may add transient metadata; the approved action and receipt
        # identity must continue to describe the original proposal.
        result = handler(action=deepcopy(action), context=context)
        if tool == "check_convergence":
            from phase_agent.analysis.convergence.normalize_convergence_result import (
                normalize_convergence_result,
            )

            result = normalize_convergence_result(result)
        execution = {"status": "completed", "tool": tool, "result": result, "error": None}
    except Exception as error:
        execution = {
            "status": "failed",
            "tool": tool,
            "result": None,
            "error": f"{type(error).__name__}: {error}",
        }
    from phase_agent.tools.dispatch.operation_outcome import operation_outcome

    execution["operation_outcome"] = operation_outcome(action, execution, context)
    if identity is not None:
        from phase_agent.tools.state.execution_receipts import record_execution_return

        from phase_agent.tools.state.execution_return_evidence import return_evidence

        record_execution_return(
            state_path,
            identity,
            execution["status"],
            evidence=return_evidence(
                execution, action=action, invocation_id=identity["invocation_id"]
            ),
        )
    return execution
