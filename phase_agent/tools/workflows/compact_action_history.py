"""Keep audit records without embedding recursive copies of the whole run state."""

from copy import deepcopy


def compact_action_history(record):
    if not isinstance(record, dict):
        return deepcopy(record)
    # Strip redundant state BEFORE copying; handler results can contain the
    # complete task ledger. Audit contents and caller ownership stay unchanged.
    compact = {key: value for key, value in record.items() if key != "state"}
    if isinstance(compact.get("result"), dict):
        compact["result"] = {k: v for k, v in compact["result"].items() if k != "state"}
    for key in ("execution", "execution_result"):
        execution = compact.get(key)
        if isinstance(execution, dict) and isinstance(execution.get("result"), dict):
            compact[key] = {
                **execution,
                "result": {k: v for k, v in execution["result"].items() if k != "state"},
            }
    return deepcopy(compact)


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
