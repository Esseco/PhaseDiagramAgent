"""通过注册表按配置调用 branch 生成策略。"""

from __future__ import annotations

from typing import Any
from fractions import Fraction

from phase_agent.science.structures.create_generation_registry import create_generation_registry
from phase_agent.science.structures.enumerate_legal_frameworks import enumerate_legal_frameworks
from phase_agent.science.structures.boundary_utils import det_H, normalize_H, normalize_fraction


def propose_branches(
    manager: Any,
    phase_references: dict[str, Any],
    *,
    quotas: dict[str, int],
    seed: int,
    parent_branch_ids=None,
    site_mappings=None,
    oxidation_states=None,
    register=True,
    registry=None,
    system_config=None,
    strategy_options=None,
    framework_enumerator=None,
    max_det_H: int | None = None,
) -> list[dict[str, Any]]:
    registry = registry or create_generation_registry()
    system = system_config or manager.data.get("system_config") or {}
    from phase_agent.configuration.schema.validate_configuration_space import (
        validate_configuration_space,
    )

    role_check = validate_configuration_space(system)
    if not role_check["valid"]:
        raise ValueError("；".join(role_check["errors"]))
    generation = system.get("generation", {})
    space = system.get("configuration_space") or {}
    roles = space.get("roles") or {}
    fixed = space.get("fixed_values") or {}
    fixed_tm = roles.get("T") == "fixed"
    if fixed_tm and space.get("fixed_T_source") != "phase_reference":
        raise ValueError("固定 T 必须指定 fixed_T_source=phase_reference")
    quotas = dict(quotas)
    from phase_agent.decisions.agent.generation_plan import configured_generation_strategies

    enabled = set(configured_generation_strategies({"system": system}))
    enabled.intersection_update(generation.get("enabled_strategies", registry.names()))
    unknown = set(quotas) - set(registry.names())
    if unknown:
        raise ValueError(f"未知策略配额：{sorted(unknown)}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in quotas.values()
    ):
        raise ValueError("配额必须是非负整数")
    disabled = {name for name, quota in quotas.items() if quota and name not in enabled}
    if disabled:
        raise ValueError(f"策略已禁用但配额非零：{sorted(disabled)}")
    if max_det_H is not None and (type(max_det_H) is not int or max_det_H <= 0):
        raise ValueError("max_det_H 必须是正整数")
    frameworks = (framework_enumerator or enumerate_legal_frameworks)(
        manager.boundary, phase_references
    )
    parents = (
        parent_branch_ids if parent_branch_ids is not None else sorted(manager.data["branches"])
    )
    if roles.get("P") == "fixed":
        frameworks = [item for item in frameworks if item["P"] == str(fixed["P"]).upper()]
    if roles.get("H") == "fixed":
        frameworks = [item for item in frameworks if item["H"] == normalize_H(fixed["H"])]
    if roles.get("x") == "fixed":
        x_fixed = normalize_fraction(fixed["x"])
        frameworks = [item for item in frameworks if x_fixed in item["allowed_x"]]
        frameworks = [{**item, "allowed_x": [x_fixed]} for item in frameworks]
    allowed_frameworks = {(item["P"], str(item["H"])) for item in frameworks}
    parents = [
        branch_id
        for branch_id in parents
        if (manager.data["branches"][branch_id]["P"], str(manager.data["branches"][branch_id]["H"]))
        in allowed_frameworks
    ]
    if max_det_H is not None:
        frameworks = [item for item in frameworks if det_H(item["H"]) <= max_det_H]
        parents = [
            branch_id
            for branch_id in parents
            if det_H(manager.data["branches"][branch_id]["H"]) <= max_det_H
        ]
        if not frameworks and any(quotas.values()):
            raise ValueError(f"det(H) ≤ {max_det_H} 下没有合法框架")
    focus = (strategy_options or {}).get("_focus")
    if focus:
        target_phase = focus["phase"]
        if target_phase != "all" and target_phase not in {row["P"] for row in frameworks}:
            raise ValueError(f"目标相 {target_phase} 不在本轮合法框架内")
        filtered = []
        for framework in frameworks:
            if focus.get("max_det_H") is not None and det_H(framework["H"]) > focus["max_det_H"]:
                continue
            if target_phase != "all" and framework["P"] != target_phase:
                continue
            xs = [
                x
                for x in framework["allowed_x"]
                if focus.get("na_min") is None
                or focus["na_min"] <= float(Fraction(normalize_fraction(x))) <= focus["na_max"]
            ]
            if xs:
                filtered.append({**framework, "allowed_x": xs})
        frameworks = filtered
        if not frameworks:
            raise ValueError("目标Na/O2范围内没有合法组分，不生成替代区域")
        target_frameworks = {(r["P"], str(r["H"])): r for r in frameworks}
        # Composition/T ordering/periodic extension require same-phase parents.
        # Competing phase deliberately retains legal parents in other phases.
        if not quotas.get("competing_phase"):
            parents = [
                p
                for p in parents
                if (manager.data["branches"][p]["P"], str(manager.data["branches"][p]["H"]))
                in target_frameworks
            ]
        allowed_frameworks = set(target_frameworks)
    context = {
        "manager": manager,
        "phase_references": phase_references,
        "frameworks": frameworks,
        "parents": parents,
    }
    common = {"site_mappings": site_mappings, "oxidation_states": oxidation_states}
    offsets = generation.get("strategy_seed_offsets", {})
    candidates = []
    for index, (name, quota) in enumerate(quotas.items()):
        if not quota:
            continue
        options = {
            **common,
            **(strategy_options or {}).get(name, {}),
            "preserve_reference_tm": fixed_tm,
        }
        produced = registry.get(name)(
            context, quota, seed + int(offsets.get(name, index * 10000)), options
        )
        for candidate in produced:
            candidate.setdefault("strategy", name)
        candidates.extend(
            candidate
            for candidate in produced
            if (max_det_H is None or det_H(candidate["H"]) <= max_det_H)
            and (candidate["P"], str(candidate["H"])) in allowed_frameworks
            and (
                not focus
                or focus.get("na_min") is None
                or focus["na_min"]
                <= float(Fraction(normalize_fraction(candidate["x"])))
                <= focus["na_max"]
            )
            and (roles.get("x") != "fixed" or normalize_fraction(candidate["x"]) == x_fixed)
        )
    for candidate in candidates:
        before = set(manager.data["branches"])
        candidate["branch_id"] = (
            manager.add_branch(
                P=candidate["P"],
                H=candidate["H"],
                x=candidate["x"],
                T=candidate["T"],
                composition=candidate.get("composition"),
            )
            if register
            else None
        )
        candidate["already_registered"] = candidate["branch_id"] in before
    return candidates
