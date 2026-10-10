"""One bounded automatic remote step under an explicit authorization snapshot."""

from phase_agent.tools.policy.automatic_mode_authorization import authorize_automatic_action
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def run_automatic_step(state_path, action, authorization, *, operation):
    state = read_json(state_path, {}) or {}
    decision = authorize_automatic_action(action, state, authorization)
    if not decision["allowed"]:
        state["automatic_mode_status"] = "paused"
        state["automatic_mode_pause_reason"] = decision["reason"]
        write_json(state_path, state)
        return {"status": "paused", "reason": decision["reason"], "state": state}
    try:
        result = operation()
    except Exception as error:
        state = read_json(state_path, state) or state
        state["automatic_mode_status"] = "paused"
        state["automatic_mode_pause_reason"] = "operation_uncertain_query_before_retry"
        state["automatic_mode_error"] = f"{type(error).__name__}: {error}"
        write_json(state_path, state)
        return {
            "status": "paused",
            "reason": "operation_uncertain_query_before_retry",
            "state": state,
        }
    state = read_json(state_path, state) or state
    usage = state.setdefault("automatic_mode_usage", {"cost": 0.0, "steps": 0})
    usage["cost"] = float(usage.get("cost", 0)) + float(decision.get("cost", 0))
    usage["steps"] = int(usage.get("steps", 0)) + 1
    state["automatic_mode_status"] = "enabled"
    write_json(state_path, state)
    return {"status": "completed", "operation": result, "state": state}


def disable_automatic_mode(state_path, *, reason="user_requested_debug_mode"):
    state = read_json(state_path, {}) or {}
    state["automatic_mode_status"] = "disabled"
    state["automatic_mode_pause_reason"] = reason
    write_json(state_path, state)
    return {"status": "debug_mode", "state": state}
