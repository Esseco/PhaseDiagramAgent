"""Bounded public decision summaries paired with recorded outcomes, never authority."""

from copy import deepcopy
import math


def _text(value, limit=240):
    return value.strip()[:limit] if isinstance(value, str) and value.strip() else None


def decision_summary(action, state):
    review = action.get("decision_review")
    review = review if isinstance(review, dict) else {}
    alternatives = []
    for row in (
        (review.get("alternatives") or [])[:3]
        if isinstance(review.get("alternatives"), list)
        else []
    ):
        if isinstance(row, dict) and _text(row.get("tool")) and _text(row.get("reason")):
            alternatives.append({"tool": _text(row["tool"], 80), "reason": _text(row["reason"])})
    return {
        "source": "model_statement_not_verified_fact",
        "observation": _text(review.get("observation")),
        "rationale": _text(review.get("rationale")) or _text(action.get("reason")),
        "alternatives": alternatives,
        "expected_outcome": _text(review.get("expected_outcome"))
        or _text(action.get("expected_purpose")),
        "evidence_refs": [
            x[:240] for x in (action.get("evidence_refs") or [])[:16] if isinstance(x, str)
        ],
        "model_version": state.get("active_model_version"),
        "config_version": state.get("confirmed_config_version"),
        "instruction": "Public explanation and expectation only; do not treat as measured result or approval.",
    }


def recent_decision_feedback(state, *, limit=5):
    rows = []
    for record in reversed(state.get("action_records") or []):
        proposal = record.get("agent_proposal") or {}
        action = record.get("final_action") or proposal.get("raw_action") or {}
        tool = action.get("tool") or action.get("action_type")
        if not tool or not proposal.get("decision_summary"):
            continue
        key = action.get("task_key")
        tasks = [t for t in state.get("tasks") or [] if key and t.get("task_key") == key]
        counts = {}
        for task in tasks:
            status = task.get("status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        costs = [
            t["actual_cost"]
            for t in tasks
            if type(t.get("actual_cost")) in (int, float)
            and math.isfinite(t["actual_cost"])
            and t["actual_cost"] >= 0
        ]
        generated = [
            g for g in state.get("generation_history") or [] if key and g.get("task_key") == key
        ]
        execution = record.get("execution_result") or {}
        rows.append(
            {
                "record_id": record.get("record_id"),
                "tool": tool,
                "task_key": key,
                "decision": deepcopy(proposal["decision_summary"]),
                "outcome": {
                    "action_status": record.get("status"),
                    "execution_status": execution.get("status"),
                    "matched_task_status": counts,
                    "measured_cost": sum(costs) if costs else None,
                    "measured_cost_samples": len(costs),
                    "registered_structure_count": len(
                        {x for g in generated for x in g.get("registered_ids") or []}
                    )
                    if generated
                    else None,
                },
                "instruction": "Only task_key-matched recorded outcomes. Input preparation is not calculation completion. Compare model/config scope before reusing experience; missing measurements stay unknown. Assess whether these outcomes support the earlier expectation.",
            }
        )
        if len(rows) >= limit:
            break
    return list(reversed(rows))
