"""Keep audit records without embedding recursive copies of the whole run state."""

from copy import deepcopy


def compact_action_history(record):
    compact = deepcopy(record)
    if not isinstance(compact, dict):
        return compact
    if isinstance(compact.get("result"), dict):
        compact["result"].pop("state", None)
    for key in ("execution", "execution_result"):
        execution = compact.get(key)
        if isinstance(execution, dict) and isinstance(execution.get("result"), dict):
            execution["result"].pop("state", None)
    compact.pop("state", None)
    return compact


def compact_state_history(state):
    """Remove redundant embedded state snapshots from old and new audit rows."""
    for key in ("action_records", "decisions", "event_history"):
        if isinstance(state.get(key), list):
            state[key] = [compact_action_history(row) for row in state[key]]
    if isinstance(state.get("invocations"), dict):
        state["invocations"] = {
            key: compact_action_history(row) for key, row in state["invocations"].items()
        }
    return state
