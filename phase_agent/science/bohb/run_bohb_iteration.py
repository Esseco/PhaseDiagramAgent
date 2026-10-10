"""生成一轮 MC 预算动作；无 evaluator 时只创建 pending 任务。"""

from copy import deepcopy

from .promote_bohb_candidates import promote_bohb_candidates
from .record_bohb_result import record_bohb_result
from .select_bohb_candidates import select_bohb_candidates
from .validate_bohb_scope import validate_bohb_scope
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost


def run_bohb_iteration(
    candidates: list[dict],
    state: dict | None,
    *,
    config: dict,
    total_mc_budget: int,
    seed: int,
    evaluator=None,
) -> dict:
    current = deepcopy(
        state
        or {"iteration": 0, "observations": [], "pending_tasks": [], "consumed_mc_budget": 0.0}
    )
    scope = validate_bohb_scope(config, candidates)
    if scope["status"] != "completed":
        return {"status": "not_configured", "state": current, "actions": [], "scope": scope}
    if current.get("scope_id") not in {None, scope["scope_id"]}:
        return {"status": "scope_changed", "state": current, "actions": [], "scope": scope}
    current["scope_id"] = scope["scope_id"]
    current["scope"] = deepcopy(config["scope"])
    by_id = {item["branch_id"]: item for item in candidates}
    levels = list(config["budget_levels"])
    pending_keys = {item["task_key"] for item in current.get("pending_tasks", [])}
    pending_pairs = {
        (item["branch_id"], item["budget"]) for item in current.get("pending_tasks", [])
    }
    proposals = promote_bohb_candidates(
        current["observations"], budget_levels=levels, eta=int(config["eta"])
    )
    if not proposals:
        available = [
            item
            for item in candidates
            if item["branch_id"]
            not in {row["branch_id"] for row in current.get("pending_tasks", [])}
        ]
        selection = select_bohb_candidates(
            available,
            current["observations"],
            count=int(config["new_candidates_per_iteration"]),
            seed=seed,
            config=config,
        )
        proposals = [
            {
                "branch_id": item["branch_id"],
                "from_budget": 0,
                "to_budget": levels[0],
                "checkpoint": None,
                "selection_mode": selection["model_mode"],
                "selection_source": selection["selection_sources"][item["branch_id"]],
            }
            for item in selection["selected"]
        ]
    actions = []
    starting_consumed = float(current.get("consumed_mc_budget", 0))
    planned = 0
    for offset, proposal in enumerate(proposals):
        branch_id, budget = proposal["branch_id"], int(proposal["to_budget"])
        increment = budget - int(proposal.get("from_budget", 0))
        task_key = f"{scope['scope_id']}:{branch_id}:{budget}:{seed + offset}"
        if (
            task_key in pending_keys
            or (branch_id, budget) in pending_pairs
            or starting_consumed + planned + increment > total_mc_budget
        ):
            continue
        cost_estimate = estimate_stage_cost(
            "deep_search",
            atom_count=by_id[branch_id].get("atom_count"),
            mc_steps=increment,
            budgets=config.get("budget_limits"),
        )
        action = {
            "task_key": task_key,
            "branch_id": branch_id,
            "stage": "deep_search",
            "status": "pending",
            "budget": budget,
            "incremental_budget": increment,
            "planned_relative_cost": cost_estimate["value"],
            "cost_estimate": cost_estimate,
            "seed": seed + offset,
            "checkpoint": proposal.get("checkpoint"),
            "selection_source": proposal.get("selection_source", "bohb_promotion"),
            "model_version": config["scope"]["mlip_version"],
            "hull_reference_version": config["scope"]["hull_reference_version"],
            "scope_id": scope["scope_id"],
        }
        action["phase_diagram_version"] = config["scope"].get("phase_diagram_version")
        actions.append(action)
        branch = by_id[branch_id]
        if branch.get("structure_id"):
            action.update(
                {
                    key: branch[key]
                    for key in (
                        "structure_id",
                        "structure_path",
                        "relaxed_ehull",
                        "relaxed_ehull_unit",
                        "ehull_source",
                        "phase_diagram_version",
                        "branch_energy_std_per_atom",
                        "hull_reference_energy_per_atom",
                    )
                }
            )
            action["max_mc_steps"] = increment
        planned += increment
        if evaluator is None:
            current.setdefault("pending_tasks", []).append(action)
            continue
        result = evaluator(branch=by_id[branch_id], action=action)
        result = {**action, **result}
        if result.get("status") in {"pending", "running"}:
            current.setdefault("pending_tasks", []).append(result)
        else:
            current = record_bohb_result(
                current,
                by_id[branch_id],
                result,
                scope_id=scope["scope_id"],
                budget=budget,
                seed=action["seed"],
                objective=config["objective"],
            )
    current["iteration"] = int(current.get("iteration", 0)) + 1
    status = (
        "budget_exhausted"
        if not actions and current.get("consumed_mc_budget", 0) >= total_mc_budget
        else "active"
    )
    reserved = sum(item.get("incremental_budget", 0) for item in current.get("pending_tasks", []))
    return {
        "status": status,
        "state": current,
        "actions": actions,
        "scope": scope,
        "remaining_mc_budget": total_mc_budget - current.get("consumed_mc_budget", 0) - reserved,
    }
