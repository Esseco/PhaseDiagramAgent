"""Keep audit records without embedding recursive copies of the whole run state."""

from copy import deepcopy


def compact_action_history(record, *, _owned=False):
    if not isinstance(record, dict):
        return deepcopy(record)
    # All execution wrappers can directly contain the canonical run state,
    # including model-refresh's execution_result.state. Strip before copying.
    compact = _audit_without_state(record, owned=_owned)
    return compact if _owned else deepcopy(compact)


def _audit_without_state(record, *, owned):
    current = record if owned else {k: v for k, v in record.items() if k != "state"}
    current.pop("state", None)
    for key in ("result", "execution", "execution_result"):
        if isinstance(current.get(key), dict):
            current[key] = _audit_without_state(current[key], owned=owned)
    return current


def compact_state_history(state, *, _owned=False):
    """Remove redundant embedded state snapshots from old and new audit rows."""
    for key in ("action_records", "decisions", "event_history"):
        if isinstance(state.get(key), list):
            state[key] = [compact_action_history(row, _owned=_owned) for row in state[key]]
    if isinstance(state.get("invocations"), dict):
        state["invocations"] = {
            key: compact_action_history(row, _owned=_owned)
            for key, row in state["invocations"].items()
        }
    return state
