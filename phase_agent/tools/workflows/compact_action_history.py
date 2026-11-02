"""Keep audit records without embedding recursive copies of the whole run state."""

from copy import deepcopy


_PREVIEW_KEYS = {
    "relax_mc_preview",
    "dft_selection_preview",
    "dft_input_preview",
    "refresh_preview",
}
_PREVIEW_SUMMARY_KEYS = {
    "task_count",
    "selected_count",
    "single_point_count",
    "relax_count",
    "qbc_available_count",
    "workload",
    "warnings",
    "mode",
    "bootstrap_first_mc",
}


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
    # Candidate pools and evidence catalogs belong to the current decision
    # context, not an immutable audit receipt.  Keep the signed raw action and
    # rationale, while dropping the repeated multi-megabyte context copy from
    # historical rows.  Pending approvals retain their complete proposal in
    # pending_execution_policies.
    for key in ("agent_proposal", "prior_proposal"):
        proposal = current.get(key)
        if isinstance(proposal, dict):
            current[key] = _compact_proposal(proposal, owned=owned)
    for key in ("result", "execution", "execution_result"):
        if isinstance(current.get(key), dict):
            current[key] = _audit_without_state(current[key], owned=owned)
    return current


def _compact_proposal(proposal, *, owned):
    """Keep approval identity while removing repeated multi-megabyte previews."""
    current = proposal if owned else dict(proposal)
    current.pop("decision_context", None)
    for key in ("action_parameters", "parameters"):
        value = current.get(key)
        if isinstance(value, dict):
            current[key] = _compact_parameters(value, owned=owned)
    raw_action = current.get("raw_action")
    if isinstance(raw_action, dict):
        raw_action = raw_action if owned else dict(raw_action)
        raw_action.pop("decision_context", None)
        parameters = raw_action.get("parameters")
        if isinstance(parameters, dict):
            raw_action["parameters"] = _compact_parameters(parameters, owned=owned)
        current["raw_action"] = raw_action
    return current


def _compact_parameters(parameters, *, owned):
    current = parameters if owned else dict(parameters)
    for key in _PREVIEW_KEYS:
        preview = current.pop(key, None)
        if isinstance(preview, dict):
            summary = {
                name: deepcopy(preview[name]) for name in _PREVIEW_SUMMARY_KEYS if name in preview
            }
            if isinstance(preview.get("selected_structures"), list):
                summary["selected_structure_count"] = len(preview["selected_structures"])
            if summary:
                current[key + "_summary"] = summary
    return current


def compact_state_history(state, *, _owned=False):
    """Remove redundant embedded state snapshots from old and new audit rows."""
    for key in ("tasks", "pending_tasks"):
        if isinstance(state.get(key), list):
            state[key] = [
                compact_action_history(row, _owned=_owned)
                if isinstance(row.get("state"), dict)
                and isinstance(row["state"].get("tasks"), list)
                and row.get("stage") is None
                and str(row.get("task_id", "")).endswith(":task")
                else row
                for row in state[key]
            ]
    for key in ("action_records", "decisions", "event_history"):
        if isinstance(state.get(key), list):
            state[key] = [compact_action_history(row, _owned=_owned) for row in state[key]]
    if isinstance(state.get("invocations"), dict):
        state["invocations"] = {
            key: compact_action_history(row, _owned=_owned)
            for key, row in state["invocations"].items()
        }
    return state


def compact_state_history_once(state, *, version=2):
    """Compact duplicate audit previews once without touching scientific facts."""
    if not isinstance(state, dict) or state.get("_history_compaction_version", 0) >= version:
        return False
    if not any(
        state.get(key) for key in ("action_records", "decisions", "event_history", "invocations")
    ):
        return False
    compact_state_history(state, _owned=True)
    state["_history_compaction_version"] = version
    return True
