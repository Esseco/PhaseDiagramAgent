"""Build the bounded, read-only state view consumed by decision agents."""

from copy import deepcopy

from phase_agent.analysis.state.build_decision_context import build_decision_context
from phase_agent.analysis.state.sanitize_untrusted_text import untrusted_text


def build_state_snapshot(state: dict, *, snapshot_index: int, config_version=None) -> dict:
    tasks = state.get("tasks") or []
    phase = state.get("phase_diagrams") or state.get("phase_diagram_state") or {}
    diagrams = phase.get("diagrams", phase) if isinstance(phase, dict) else {}
    qbc_rows = state.get("qbc_candidates") or []
    uncertainty = [_qbc_summary(row) for row in qbc_rows[-50:]]
    uncertainty = [row for row in uncertainty if row]
    actions = state.get("action_records") or state.get("decisions") or []
    rewards = state.get("rewards") or []
    failed = [
        item
        for item in tasks
        if item.get("status") in {"failed", "timeout", "unknown", "not_configured"}
    ]
    available = state.get("budget_remaining", state.get("remaining_budget"))
    snapshot = {
        "schema_version": 1,
        "snapshot_id": f"state-{snapshot_index:06d}",
        "snapshot_index": snapshot_index,
        "config_version": config_version or state.get("confirmed_config_version"),
        "current_status": state.get("status", "ready"),
        "current_convex_hull": _hull_summary(diagrams),
        "search_coverage": deepcopy(state.get("coverage") or state.get("coverage_report") or {}),
        "available_budget": deepcopy(available),
        "budget_usage": deepcopy(state.get("budget_usage") or {}),
        "uncertainty": uncertainty,
        "qbc_uncertainty": uncertainty,
        "mlip_status": {
            "active_version": state.get("active_model_version"),
            "validation": deepcopy(state.get("mlip_validation") or {}),
            "stale_results": deepcopy(state.get("stale_model_dependent_results") or []),
        },
        "search_history": [_action_summary(row) for row in actions[-10:]],
        "recent_action_history": [_action_summary(row) for row in actions[-10:]],
        "short_term_memory": {
            "recent_rewards": deepcopy(rewards[-10:]),
            "failed_tasks": [_task_summary(item) for item in failed[-10:]],
            "task_status": _task_counts(tasks),
            "pending_branch_screening": deepcopy(state.get("pending_branch_screening")),
        },
        "long_term_memory": deepcopy(
            (state.get("decision_memory") or {}).get("long_term")
            or {
                "human_system_knowledge": deepcopy(
                    (state.get("decision_memory") or {}).get("long_term_advice") or []
                ),
                "physical_priors": [],
                "frozen_parameter_advice": [],
                "search_rules": [],
            }
        ),
        "available_branches": deepcopy(state.get("branch_candidates") or []),
    }
    snapshot["decision_context"] = build_decision_context(
        {**state, "decision_memory": state.get("decision_memory") or {}}
    )
    return snapshot


def _hull_summary(diagrams):
    output = {}
    for method in ("mlip", "dft"):
        item = diagrams.get(method) or {}
        entries = item.get("entries") or []
        output[method] = {
            "status": item.get("status"),
            "version": item.get("version"),
            "energy_basis_id": item.get("energy_basis_id"),
            "stable_entries": [
                {
                    key: deepcopy(row.get(key))
                    for key in ("record_id", "composition", "ehull", "energy")
                }
                for row in entries
                if row.get("is_stable")
            ],
        }
    return output


def _qbc_summary(row):
    qbc = row.get("qbc") or row
    values = {
        key: deepcopy(qbc.get(key))
        for key in ("f_std_max", "f_std_p95", "force_rms_disagreement", "energy_per_atom_std")
        if qbc.get(key) is not None
    }
    if not values:
        return None
    return {"candidate_id": row.get("candidate_id") or row.get("structure_id"), **values}


def _action_summary(row):
    action = row.get("final_action") or row.get("action") or {}
    return {
        "record_id": row.get("record_id"),
        "status": row.get("status"),
        "action_type": action.get("action_type") or action.get("tool"),
        "reason": action.get("reason"),
    }


def _task_summary(item):
    return {
        **{key: deepcopy(item.get(key)) for key in ("task_id", "task_key", "stage", "status")},
        "error": untrusted_text(item.get("error") or item.get("failure_reason")),
    }


def _task_counts(tasks):
    counts = {}
    for task in tasks:
        status = task.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts
