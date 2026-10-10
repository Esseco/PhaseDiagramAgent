"""Estimate downstream work from parent-cell metadata BEFORE expansion."""

from fractions import Fraction
from math import isfinite

from phase_agent.science.structures.boundary_utils import (
    allowed_phases,
    allowed_phases_at_x,
    allowed_supercells,
    det_H,
    load_structure,
)
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.analysis.cost.predict_runtime import predict_runtime


def generation_framework_costs(manager, phase_references, config, state):
    budgets = config.get("budgets") or {}
    limits = budgets.get("structure_limits") or {}
    policy = config.get("mc_policy") or {}
    steps = [
        r.get("max_mc_steps")
        for r in [*(policy.get("tiers") or []), *(policy.get("initial_ehull_bands") or [])]
    ]
    steps += [policy.get("exploration_max_mc_steps")]
    steps = [v for v in steps if type(v) is int and v > 0]
    mc_steps = max(steps) if steps else None
    # Approval may request more initial states than the default; price the supported ceiling.
    initial = 3
    boundary = manager.boundary
    output = []
    for phase in sorted(allowed_phases(boundary["P"])):
        if phase not in phase_references:
            raise ValueError(f"预估缺少母结构：{phase}")
        reference = load_structure(phase_references[phase])
        oxygen = sum(s.is_ordered and s.specie.symbol == "O" for s in reference)
        sodium = sum(s.is_ordered and s.specie.symbol == "Na" for s in reference)
        if oxygen <= 0:
            raise ValueError("生成预估需要含氧母结构")
        for matrix in allowed_supercells(boundary["H"], phase):
            size = det_H(matrix)
            atoms = len(reference) * size
            if limits.get("max_atoms") is not None and atoms > limits["max_atoms"]:
                continue
            if limits.get("max_det_H") is not None and size > limits["max_det_H"]:
                continue
            xs = [
                float(Fraction(2 * n, oxygen * size))
                for n in range(sodium * size + 1)
                if Fraction(2 * n, oxygen * size) <= 1
                and phase in allowed_phases_at_x(boundary["P"], Fraction(2 * n, oxygen * size))
            ]
            if not xs:
                continue
            costs, times = {}, {}
            for stage in ("relax_and_feature", "deep_search"):
                if stage == "deep_search" and mc_steps is None:
                    costs[stage] = times[stage] = None
                    continue
                count = initial if stage == "relax_and_feature" else 1
                costs[stage] = estimate_stage_cost(
                    stage,
                    atom_count=atoms,
                    initial_state_count=count,
                    mc_steps=mc_steps if stage == "deep_search" else None,
                    budgets=budgets,
                )["value"]
                timing = predict_runtime(
                    stage,
                    atom_count=atoms,
                    state=state,
                    mc_steps=mc_steps if stage == "deep_search" else None,
                )
                seconds = timing.get("elapsed_seconds")
                times[stage] = seconds * count if seconds is not None else None
            output.append(
                {
                    "phase": phase,
                    "H": matrix,
                    "det_H": size,
                    "atom_count_upper": atoms,
                    "allowed_x": xs,
                    "initial_states": initial,
                    "mc_steps_upper": mc_steps,
                    "stage_costs": costs,
                    "stage_seconds": times,
                }
            )
    return output


def attach_generation_preflight(action, state, context, config):
    if action.get("tool") != "generate_branches" or not (action.get("parameters") or {}).get(
        "generation_plan"
    ):
        return action
    manager = (context or {}).get("manager")
    if manager is None:
        raise ValueError("缺少账本，不能在扩胞前估算后续成本")
    frameworks = generation_framework_costs(
        manager, (context or {}).get("phase_references") or {}, config, state
    )
    preview = preview_generation_cost(action, frameworks, state, config)
    if preview["status"] == "over_budget":
        raise ValueError(
            f"扩胞前Relax+MC粗估 {preview['relax_mc_cost_upper']:.3g} 超过剩余预算 {preview['remaining_budget']:.3g}；请缩小胞/数量或修订预算，未生成结构"
        )
    action.setdefault("parameters", {})["generation_cost_preview"] = preview
    return action


def preview_generation_cost(action, frameworks, state, config):
    params = action.get("parameters") or {}
    plan = params.get("generation_plan")
    if not plan:
        return {"status": "unavailable", "reason": "需指定生成分配后预估后续成本"}
    rows, envelopes = [], []
    for allocation in plan:
        caps = [v for v in (allocation.get("max_det_H"), params.get("max_det_H")) if v is not None]
        cap = min(caps) if caps else None
        eligible = [
            f
            for f in frameworks
            if (allocation["phase"] == "all" or f["phase"] == allocation["phase"])
            and (cap is None or f["det_H"] <= cap)
            and (
                allocation.get("na_min") is None
                or any(allocation["na_min"] <= x <= allocation["na_max"] for x in f["allowed_x"])
            )
        ]
        if not eligible:
            raise ValueError("生成分配没有符合成本/相/Na/超胞限制的合法框架")
        if any(f["stage_costs"]["deep_search"] is None for f in eligible):
            raise ValueError("缺少已确认MC步数，不能预估扩胞后续成本")
        per_branch = max(
            f["stage_costs"]["relax_and_feature"] + f["stage_costs"]["deep_search"]
            for f in eligible
        )
        timings = [
            f["stage_seconds"]["relax_and_feature"] + f["stage_seconds"]["deep_search"]
            for f in eligible
            if all(f["stage_seconds"][s] is not None for s in ("relax_and_feature", "deep_search"))
        ]
        seconds = max(timings) if len(timings) == len(eligible) else None
        rows.append(
            {
                "strategy": allocation["strategy"],
                "phase": allocation["phase"],
                "atom_count_upper": max(f["atom_count_upper"] for f in eligible),
                "relax_mc_per_branch_upper": per_branch,
                "candidate_quota": allocation["quota"],
            }
        )
        envelopes.extend([(per_branch, seconds)] * allocation["quota"])
    selected = sorted(envelopes, key=lambda row: row[0], reverse=True)[: params["batch_size"]]
    cost = sum(row[0] for row in selected)
    limits = config.get("budgets") or {}
    used = float((state.get("budget_usage") or {}).get("total_relative_cost", 0))
    remaining = (
        max(
            0.0,
            float(limits["total_relative_cost"])
            - used
            - float(state.get("reserved_relative_cost", 0)),
        )
        if limits.get("total_relative_cost") is not None
        else None
    )
    if not isfinite(cost):
        raise ValueError("扩胞成本预估无效")
    return {
        "status": "within_budget" if remaining is None or cost <= remaining else "over_budget",
        "allocations": rows,
        "selected_branch_upper": len(selected),
        "relax_mc_cost_upper": cost,
        "remaining_budget": remaining,
        "serial_seconds_upper": sum(r[1] for r in selected)
        if all(r[1] is not None for r in selected)
        else None,
        "assumption": "每个入选branch最多3个Relax初态、1个MC按确认策略最大步数；Na满占原子数上界；不含第二段MC或排队；DFT不属于本次branch规划。",
    }
