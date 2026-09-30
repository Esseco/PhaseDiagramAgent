"""Preview the existing MC policy against a frozen Relax hull before approval."""
import hashlib
import json

from analysis_layer.phase.branch_relax_hull import rank_relaxed_branches
from execution_layer.budget.estimate_stage_cost import estimate_stage_cost
from scientific_layer.mc.schedule_tiered_mc import schedule_tiered_mc
from scientific_layer.mc.second_round_state import (
    first_round_source, reconciled_mc_state, second_round_candidates,
)


def estimate_branch_mc_budget(candidates, pool, state, config, *, step_limit, seed):
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if (diagram.get("status") != "completed" or diagram.get("model_version") != pool.get("model_version")
            or not diagram.get("version")):
        raise ValueError("当前 MLIP 相图未就绪，不能按 Ehull/atom 预估 MC")
    ranked, missing = rank_relaxed_branches(candidates, pool,
        uncertainty_weight=float((config.get("bohb") or {}).get("uncertainty_weight", 1.0)),
        phase_diagram=diagram)
    source = first_round_source(state, pool["model_version"])
    if source:
        ranked, missing_mc = second_round_candidates(ranked, state, diagram, pool["model_version"])
        missing = sorted(set(missing + missing_mc))
    tier_state = reconciled_mc_state(state)
    full = schedule_tiered_mc(ranked, tier_state,
        policy=config["mc_policy"], total_budget=float("inf"), seed=seed,
        model_version=pool["model_version"], hull_reference_version=pool["version"],
        phase_diagram_version=diagram["version"])
    full_steps = sum(row["max_mc_steps"] for row in full["actions"])
    full_cost = sum(estimate_stage_cost("deep_search", atom_count=next(
        candidate["atom_count"] for candidate in ranked if candidate["branch_id"] == row["branch_id"]),
        mc_steps=row["max_mc_steps"], budgets=config["budgets"])["value"] for row in full["actions"])
    scheduled = schedule_tiered_mc(ranked, tier_state,
        policy=config["mc_policy"], total_budget=step_limit, seed=seed,
        model_version=pool["model_version"], hull_reference_version=pool["version"],
        phase_diagram_version=diagram["version"])
    rows = []
    for task in scheduled["actions"]:
        candidate = next(row for row in ranked if row["branch_id"] == task["branch_id"])
        cost = estimate_stage_cost("deep_search", atom_count=candidate["atom_count"],
            mc_steps=task["max_mc_steps"], budgets=config["budgets"])
        rows.append({**task, "atom_count": candidate["atom_count"],
                     "planned_relative_cost": cost["value"], "cost_basis": cost})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    return {"hull_reference_version": pool["version"], "model_version": pool["model_version"],
            "round_kind": "second" if source else "first",
            "first_round_source_checksum": source["checksum"] if source else None,
            "first_round_completed_count": source["completed_count"] if source else 0,
            "phase_diagram_version": diagram["version"],
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
