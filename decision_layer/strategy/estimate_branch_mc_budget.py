"""Preview the existing MC policy against a frozen Relax hull before approval."""
import hashlib
import json

from analysis_layer.phase.branch_relax_hull import rank_relaxed_branches
from execution_layer.budget.estimate_stage_cost import estimate_stage_cost
from scientific_layer.mc.schedule_tiered_mc import schedule_tiered_mc


def estimate_branch_mc_budget(candidates, pool, state, config, *, step_limit, seed):
    ranked, missing = rank_relaxed_branches(candidates, pool,
        uncertainty_weight=float((config.get("bohb") or {}).get("uncertainty_weight", 1.0)))
    full = schedule_tiered_mc(ranked, state.get("tiered_mc_state") or {},
        policy=config["mc_policy"], total_budget=float("inf"), seed=seed,
        model_version=pool["model_version"], hull_reference_version=pool["version"])
    full_steps = sum(row["max_mc_steps"] for row in full["actions"])
    full_cost = sum(estimate_stage_cost("deep_search", atom_count=next(
        candidate["atom_count"] for candidate in ranked if candidate["branch_id"] == row["branch_id"]),
        mc_steps=row["max_mc_steps"], budgets=config["budgets"])["value"] for row in full["actions"])
    scheduled = schedule_tiered_mc(ranked, state.get("tiered_mc_state") or {},
        policy=config["mc_policy"], total_budget=step_limit, seed=seed,
        model_version=pool["model_version"], hull_reference_version=pool["version"])
    rows = []
    for task in scheduled["actions"]:
        candidate = next(row for row in ranked if row["branch_id"] == task["branch_id"])
        cost = estimate_stage_cost("deep_search", atom_count=candidate["atom_count"],
            mc_steps=task["max_mc_steps"], budgets=config["budgets"])
        rows.append({**task, "atom_count": candidate["atom_count"],
                     "planned_relative_cost": cost["value"], "cost_basis": cost})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    return {"hull_reference_version": pool["version"], "model_version": pool["model_version"],
            "step_limit": step_limit, "requested_steps": sum(row["max_mc_steps"] for row in rows),
            "target_steps": step_limit, "full_plan_steps": full_steps,
            "full_plan_relative_cost": full_cost, "full_plan_branch_count": len(full["actions"]),
            "exceeds_target": full_steps > step_limit,
            "compression_options": (["run_full_after_budget_review", "reduce_branch_count", "reduce_steps"]
                                    if full_steps > step_limit else []),
            "compression_requires_user_choice": full_steps > step_limit,
            "estimated_relative_cost": sum(row["planned_relative_cost"] for row in rows),
            "selected_branch_count": len(rows), "candidate_count": len(candidates),
            "missing_branch_ids": missing, "excluded_candidates": scheduled.get("excluded_candidates", []),
            "allocation_checksum": digest, "allocations": rows,
            "energy_unit": "eV/atom", "status": scheduled["status"]}
