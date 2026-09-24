"""通过已有模块或显式适配器执行已校验动作。"""


def execute_agent_action(action: dict, *, handlers: dict | None = None) -> dict:
    if action.get("action") == "wait":
        return {"status": "deferred", "action": action, "result": None}
    handler = (handlers or {}).get(action.get("action"))
    if handler is None:
        return {"status": "not_configured", "action": action, "result": None, "error": f"no handler for {action.get('action')}"}
    try:
        return {"status": "completed", "action": action, "result": handler(action), "error": None}
    except Exception as error:
        return {"status": "failed", "action": action, "result": None, "error": f"{type(error).__name__}: {error}"}
