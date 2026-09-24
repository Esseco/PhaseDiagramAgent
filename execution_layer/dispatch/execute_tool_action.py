"""Execute a validated registered tool or return an explicit adapter status."""


def execute_tool_action(action: dict, *, registry: dict, context: dict) -> dict:
    item = registry.get(action.get("tool")) or {}; handler = item.get("handler")
    if handler is None:
        return {"status": "not_configured", "tool": action.get("tool"), "result": None, "error": "tool handler not configured"}
    try:
        result = handler(action=action, context=context)
        if action["tool"] == "check_convergence":
            from analysis_layer.convergence.normalize_convergence_result import normalize_convergence_result
            result = normalize_convergence_result(result)
        return {"status": "completed", "tool": action["tool"], "result": result, "error": None}
    except Exception as error:
        return {"status": "failed", "tool": action["tool"], "result": None, "error": f"{type(error).__name__}: {error}"}
