"""Marginal-budget evidence and explicit decision/task outcome attribution."""

from copy import deepcopy
from collections import defaultdict
import math

TERMINAL = {"completed", "failed", "timeout", "cancelled"}


def numeric_cost(value):
    if isinstance(value, dict):
        if value.get("unit", "relative_cost") != "relative_cost":
            return None
        value = value.get("value", value.get("relative_cost"))
    return (
        float(value)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


def budget_decision_outcomes(state, *, limit=None):
    """Attribute exact links; optional limit selects decisions, never their evidence."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    decisions = [
        row
        for row in state.get("action_records") or []
        if isinstance((row.get("final_action") or {}).get("round_budget_review"), dict)
    ]
    if limit is not None:
        decisions = decisions[-limit:]
    if not decisions:
        return []
    tasks = state.get("tasks") or []
    rewards = state.get("rewards") or []
    by_task, by_parent, rewards_by_task = defaultdict(list), defaultdict(list), defaultdict(list)
    for position, task in enumerate(tasks):
        if task.get("task_id"):
            by_task[task["task_id"]].append(position)
            if task.get("parent_decision_id"):
                by_parent[task["parent_decision_id"]].append(position)
    for position, reward in enumerate(rewards):
        for task_id in set(reward.get("task_ids") or []):
            rewards_by_task[task_id].append(position)
    output = []
    for decision in decisions:
        action = decision.get("final_action") or {}
        review = action.get("round_budget_review")
        if not isinstance(review, dict):
            continue
        identity = decision.get("record_id")
        result = (decision.get("execution_result") or {}).get("result") or {}
        explicit = (
            {row.get("task_id") for row in result.get("tasks") or [] if isinstance(row, dict)}
            if isinstance(result, dict)
            else set()
        )
        positions = set(by_parent.get(identity, []) if identity else [])
        for task_id in explicit:
            positions.update(by_task.get(task_id, []))
        # Positions deduplicate overlapping links, preserving duplicate source rows and task order.
        linked = [tasks[position] for position in sorted(positions)]
        ids = {task["task_id"] for task in linked}
        versions = {task.get("model_version") for task in linked}
        version = next(iter(versions)) if len(versions) == 1 else None
        costs = [numeric_cost(task.get("actual_cost")) for task in linked]
        done = bool(linked) and all(task.get("status") in TERMINAL for task in linked)
        actual_cost = sum(costs) if done and all(cost is not None for cost in costs) else None
        reward_positions = {
            position for task_id in ids for position in rewards_by_task.get(task_id, [])
        }
        attributable = [rewards[position] for position in sorted(reward_positions)]
        output.append(
            {
                "decision_id": identity,
                "selected_path": review.get("choice"),
                "prediction": deepcopy(review),
                "model_version": version,
                "config_version": decision.get(
                    "config_version", state.get("confirmed_config_version")
                ),
                "task_ids": sorted(ids),
                "task_count": len(linked),
                "status": "complete" if done else "partial" if linked else "awaiting_task_links",
                "completed_count": sum(task.get("status") == "completed" for task in linked),
                "actual_relative_cost": actual_cost,
                "cost_status": "measured" if actual_cost is not None else "unknown_or_incomplete",
                "observed_rewards": [
                    {
                        key: deepcopy(reward.get(key))
                        for key in (
                            "batch_id",
                            "reward",
                            "new_stable_entries",
                            "ehull_improvement",
                            "energy_method",
                            "energy_basis_id",
                            "previous_version",
                            "current_version",
                            "model_version",
                        )
                    }
                    for reward in attributable
                ],
                "comparison_status": "versioned_observations_only_not_counterfactual",
                "unselected_path_actual_benefit": None,
            }
        )
    return output


def round_budget_evidence(state):
    from phase_agent.evidence.budget_experience import budget_experience_context

    reports = state.get("training_result_reports")
    if reports is None:
        from phase_agent.evidence.training_result_evidence import training_result_evidence

        reports = training_result_evidence(state)
    candidates = state.get("qbc_candidates") or []
    labelled = {
        row.get("structure_id")
        for row in state.get("dft_training_records") or []
        if row.get("training_ready") is True and row.get("checks_passed") is True
    }
    remaining = [
        row
        for row in candidates
        if (row.get("structure_id") or row.get("candidate_id")) not in labelled
        and not row.get("duplicate_of")
    ]
    outcomes = budget_decision_outcomes(state, limit=8)
    required = bool(
        reports or state.get("dft_result_exports") or state.get("post_dft_decided_rounds")
    )
    return {
        "required": required,
        "active_model_version": state.get("active_model_version"),
        "training_reports": deepcopy(reports[-3:]),
        "existing_pool": {
            "count": len(candidates),
            "unlabelled_candidates": len(remaining),
            "qbc_available_count": sum(
                (row.get("qbc") or {}).get("status") == "completed" for row in remaining
            ),
            "qbc_model_versions": sorted(
                {str(row.get("model_version")) for row in remaining if row.get("model_version")}
            ),
            "instruction": "Counts alone do not establish representativeness; unknown or old-model QBC must be refreshed.",
        },
        "budget_remaining": deepcopy(state.get("budget_remaining")),
        "observed_budget_outcomes": deepcopy(outcomes),
        "budget_experience": budget_experience_context(state),
        "instruction": "Compare supplement_dft from existing candidates versus new_search, using marginal downstream Relax/MC/DFT/training costs, hull/coverage gains and model risk. Historical sunk cost is evidence, not new expenditure. Missing costs/gains remain unknown. Do not compare raw rewards across model or energy-basis versions. Only the chosen path has observed outcomes; alternatives remain hypotheses. Coverage and error are evidence for expected benefit and uncertainty, NEVER hard gates for new_search. New search may win even with large error or incomplete coverage. Compare complete paths through downstream DFT and any retraining; no fixed accuracy threshold or DFT-first rule.",
    }
