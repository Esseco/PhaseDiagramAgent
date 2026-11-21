"""Reuse weighted sampling with explicit phase/Na coverage and per-round caps."""

from copy import deepcopy
from phase_agent.agents.reliability.selection.select_dft_candidates import select_dft_candidates
from phase_agent.tools.budget.estimate_stage_cost import estimate_dft_cost
from phase_agent.agents.reliability.selection.dft_round_policy import (
    dft_task_limit,
    round_dft_tasks,
)


def recommend_post_mc_dft(candidates, config, state=None):
    pool = deepcopy(candidates)
    state = state or {}
    prior = round_dft_tasks(state, config)
    remaining_count = dft_task_limit(config)
    remaining_cost = None
    done = {row.get("structure_id") for row in prior}
    remote_sampling = state.get("remote_dft_sampling") or {}
    if remote_sampling:
        latest = list(remote_sampling.values())[-1]
        if latest.get("status") == "accepted":
            qbc_cfg = config.get("qbc") or {}
            selected = [
                {
                    **deepcopy(row),
                    "dft_selection_mode": row.get("selection_mode", "remote_kmeans_pp"),
                    "dft_selection_score": row.get("weight"),
                    "estimated_cost": (
                        estimate_dft_cost("DFT_SINGLE_POINT", row, qbc_cfg)
                        if qbc_cfg.get("action_costs")
                        else 0.0
                    ),
                }
                for row in latest.get("selected") or []
                if row.get("structure_id") not in done
            ][:remaining_count]
            return {
                "selected_candidates": selected,
                "rejected_candidates": [],
                "summary": {
                    "selected": len(selected),
                    "gs_count": latest.get("gs_count", 0),
                    "kmeans_count": latest.get("kmeans_count", 0),
                    "source": "remote_dft_sampling",
                    "relative_cost": sum(r["estimated_cost"] for r in selected),
                },
            }
    pool = [row for row in pool if row.get("candidate_id") not in done]
    if remaining_count <= 0:
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
