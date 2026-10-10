"""Reuse weighted sampling with explicit phase/Na coverage and per-round caps."""

from copy import deepcopy
from phase_agent.decisions.qbc_selection.select_dft_candidates import select_dft_candidates
from phase_agent.tools.budget.estimate_stage_cost import estimate_dft_cost


def recommend_post_mc_dft(candidates, config, state=None):
    selection = (config.get("dft") or {}).get("selection") or {}
    pool = deepcopy(candidates)
    state = state or {}
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name") or state.get("active_model_version")
    prior = [
        row
        for row in state.get("tasks") or []
        if row.get("stage") == "dft_single_point"
        and row.get("model_version") == version
        and row.get("generation_cycle") == len(state.get("generation_history") or [])
        and row.get("status") != "cancelled"
    ]
    remaining_count = int(selection.get("single_point_max_per_round", 100)) - len(prior)
    remaining_cost = float(selection.get("single_point_cost_per_round", 5000)) - sum(
        float(row.get("planned_relative_cost") or 0) for row in prior
    )
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
    if limit.get("max_tasks") is not None:
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
    if remaining_count <= 0 or remaining_cost <= 0:
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
