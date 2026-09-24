"""在相同候选、随机种子和 MC 总预算下比较三种策略。"""

from copy import deepcopy

from scientific_layer.bohb.record_bohb_result import record_bohb_result
from scientific_layer.bohb.run_bohb_iteration import run_bohb_iteration
from .run_random_hyperband import run_random_hyperband
from scientific_layer.bohb.validate_bohb_scope import validate_bohb_scope


def compare_bohb_methods(candidates: list[dict], *, config: dict, total_mc_budget: int, seed: int, simulator, metric_evaluator=None) -> dict:
    results = {}
    scope = validate_bohb_scope(config, candidates)
    if scope["status"] != "completed":
        return {"status": "not_configured", "scope": scope}
    fixed = {"scope_id": scope["scope_id"], "scope": deepcopy(config["scope"]), "observations": [], "pending_tasks": [], "consumed_mc_budget": 0, "iteration": 0}
    maximum = max(config["budget_levels"])
    for offset, branch in enumerate(candidates):
        if fixed["consumed_mc_budget"] + maximum > total_mc_budget:
            break
        action = {"budget": maximum, "incremental_budget": maximum, "seed": seed + offset, "task_key": f"fixed:{branch['branch_id']}:{seed + offset}", "model_version": config["scope"]["mlip_version"], "hull_reference_version": config["scope"]["hull_reference_version"]}
        fixed = record_bohb_result(fixed, branch, {**action, **simulator(branch=branch, action=action)}, scope_id=scope["scope_id"], budget=maximum, seed=seed + offset, objective=config["objective"])
    fixed_summary = _summary(fixed, candidates, metric_evaluator, maximum)
    results["fixed_budget"] = fixed_summary

    hyperband_state = run_random_hyperband(candidates, config=config, total_mc_budget=total_mc_budget, seed=seed, evaluator=simulator)
    random_summary = _summary(hyperband_state, candidates, metric_evaluator, maximum)
    results["random_hyperband"] = random_summary
    bohb_ready = all(item.get("bohb_features") is not None for item in candidates)
    bohb_summary = (_run_method(candidates, config, total_mc_budget, seed, simulator, metric_evaluator)
                    if bohb_ready else {"status": "insufficient_features"})
    results["bohb"] = bohb_summary
    results["random"] = random_summary
    results["default_relax_hull_tiered_mc"] = fixed_summary
    results["bohb_variant"] = bohb_summary
    comparable = all((results[name].get("hull_quality") is not None)
                     for name in ("random", "default_relax_hull_tiered_mc", "bohb_variant"))
    return {"status": "completed", "initial_candidate_ids": [item["branch_id"] for item in candidates], "total_mc_budget": total_mc_budget, "seed": seed, "results": results,
            "comparison_conclusion": "metrics_available" if comparable else "insufficient_data_cannot_determine_benefit",
            "controls": {"same_candidate_pool": True, "same_initial_seed": True,
                         "same_total_mc_budget": True},
            "warning": "Offline replay does not prove BOHB superiority; BOHB uses only decision-time features."}


def _run_method(candidates, config, total, seed, simulator, metric_evaluator):
    state = None
    for iteration in range(100):
        output = run_bohb_iteration(candidates, state, config=config, total_mc_budget=total, seed=seed + iteration * 1000, evaluator=simulator)
        state = output["state"]
        if not output["actions"] or state["consumed_mc_budget"] >= total:
            break
    return _summary(state, candidates, metric_evaluator, max(config["budget_levels"]))


def _summary(state, candidates, metric_evaluator, important_budget):
    completed = [item for item in state.get("observations", []) if item.get("status") == "completed"]
    best = {}
    for item in completed:
        group = item.get("group", "unknown")
        best[group] = min(best.get(group, float("inf")), item["loss"])
    important = {item["branch_id"] for item in candidates if item.get("important")}
    evaluated = {item["branch_id"] for item in completed if item.get("budget") == important_budget}
    metrics = metric_evaluator(state=state, candidates=candidates) if metric_evaluator else {"hull_quality": None}
    return {"state": state, "completed_evaluations": len(completed), "actual_total_cost": state.get("consumed_mc_budget", 0), "best_loss_by_group": best, "important_high_budget_missed": sorted(important - evaluated), **metrics}
