"""Reuse weighted sampling with explicit phase/Na coverage and per-round caps."""

from copy import deepcopy
from phase_agent.decisions.qbc_selection.select_dft_candidates import select_dft_candidates
from phase_agent.tools.budget.estimate_stage_cost import estimate_dft_cost
from phase_agent.decisions.qbc_selection.dft_round_policy import dft_task_limit, round_dft_tasks


def recommend_post_mc_dft(candidates, config, state=None):
    pool = deepcopy(candidates)
    state = state or {}
    prior = round_dft_tasks(state, config)
    from phase_agent.tools.budget.check_budget import budget_is_hard

    hard = budget_is_hard(config.get("budgets"))
    remaining_count = dft_task_limit(config) - len(prior)
    remaining_cost = float("inf")
    done = {row.get("structure_id") for row in prior}
    pool = [row for row in pool if row.get("candidate_id") not in done]
    budgets = config.get("budgets") or {}
    limit = (budgets.get("stage_limits") or {}).get("dft_single_point") or {}
    used = ((state.get("budget_usage") or {}).get("stages") or {}).get("dft_single_point") or {}
    reservations = [
        row
        for row in (state.get("budget_reservations") or {}).values()
        if row.get("status") in {"reserved", "submitted", "running"}
    ]
    stage_reserved = [row for row in reservations if row.get("stage") == "dft_single_point"]
    if hard and limit.get("max_tasks") is not None:
        remaining_count = min(
            remaining_count,
            int(limit["max_tasks"]) - int(used.get("tasks") or 0) - len(stage_reserved),
        )
    if limit.get("max_cost") is not None:
        remaining_cost = min(
            remaining_cost,
            float(limit["max_cost"])
            - float(used.get("cost") or 0)
            - sum(float(row.get("reserved_cost") or 0) for row in stage_reserved),
        )
    if budgets.get("total_relative_cost") is not None:
        remaining_cost = min(
            remaining_cost,
            float(budgets["total_relative_cost"])
            - float((state.get("budget_usage") or {}).get("total_relative_cost") or 0)
            - sum(float(row.get("reserved_cost") or 0) for row in reservations),
        )
    if not hard:
        remaining_cost = None
    if remaining_count <= 0 or (remaining_cost is not None and remaining_cost <= 0):
        return {"selected_candidates": [], "summary": {"selected": 0, "relative_cost": 0}}
    groups = [(row.get("phase"), row.get("x_Na_per_O2")) for row in pool]
    qbc_config = config.get("qbc") or {}
    for row, group in zip(pool, groups):
        row["diversity_score"] = 1.0 / groups.count(group)
        row["estimated_cost"] = estimate_dft_cost("DFT_SINGLE_POINT", row, qbc_config)
    result = select_dft_candidates(
        pool,
        batch_size=remaining_count,
        cost_budget=remaining_cost,
        seed=0,
        cover_phases=True,
        audit_fraction=float((config.get("dft") or {}).get("independent_audit_fraction", 0.05)),
    )
    return result
