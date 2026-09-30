"""统一相对成本估算；基准参数需用实际同后端任务校准。"""

import math


def estimate_stage_cost(stage, *, atom_count=None, initial_state_count=1, mc_steps=None, budgets=None):
    if budgets is None:
        from config_layer.defaults.default_budget_rules import default_budget_rules
        budgets = default_budget_rules()
    rules = budgets
    model = rules.get("cost_model", {})
    reference = float(model.get("reference_atoms", 40))
    atoms = reference if atom_count is None else float(atom_count)
    exponent = float(model.get("stage_exponents", {}).get(stage, model.get("atom_exponent", 1)))
    base = float(rules["stage_limits"][stage]["task_cost"])
    initial = float(initial_state_count)
    scale = float(model.get("scale", 1))
    values = [reference, atoms, initial, exponent, base, scale]
    if not all(math.isfinite(v) for v in values) or min(reference, atoms, initial) <= 0 or min(exponent, base, scale) < 0:
        raise ValueError("成本参数必须有限且规模为正，成本及指数非负")
    search_factor = 1.0
    if stage == "deep_search":
        reference_steps = float(model.get("reference_mc_steps", 1))
        step_cost_factor = float(model.get("mc_step_cost_factor", 0.1))
        steps = 1.0 if mc_steps is None else float(mc_steps)
        if (not math.isfinite(steps) or steps < 0 or not math.isfinite(reference_steps)
                or reference_steps <= 0 or not math.isfinite(step_cost_factor)
                or step_cost_factor < 0):
            raise ValueError("MC 步数和成本系数必须有限非负，参考步数必须为正")
        search_factor = steps / reference_steps * step_cost_factor
    return {"value": base * (atoms / reference) ** exponent * initial * search_factor * scale,
            "unit": "relative_cost", "basis": "reference_size_assumed" if atom_count is None else "size_aware_estimate",
            "reference_atoms": reference, "atom_count": atoms, "stage_factor": base,
            "size_factor": (atoms / reference) ** exponent, "search_factor": search_factor,
            "initial_state_count": initial, "wall_time": None}


def estimate_dft_cost(action, metric, config):
    if action not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
        return 0.0
    stage = "dft_single_point" if action == "DFT_SINGLE_POINT" else "dft_relax"
    # 兼容显式历史 action_costs，但所有金额均为相同相对单位。
    rules = {"stage_limits": {stage: {"task_cost": config["action_costs"][action]}}, "cost_model": config.get("cost_model", {})}
    return estimate_stage_cost(stage, atom_count=metric.get("atom_count"), budgets=rules)["value"]
