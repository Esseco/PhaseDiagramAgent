"""通过注册表按配置调用 branch 生成策略。"""

from __future__ import annotations

from typing import Any

from scientific_layer.structures.create_generation_registry import create_generation_registry
from scientific_layer.structures.enumerate_legal_frameworks import enumerate_legal_frameworks


def propose_branches(manager: Any, phase_references: dict[str, Any], *, quotas: dict[str, int], seed: int, parent_branch_ids=None, site_mappings=None, oxidation_states=None, register=True, registry=None, system_config=None, strategy_options=None, framework_enumerator=None) -> list[dict[str, Any]]:
    registry = registry or create_generation_registry()
    generation = (system_config or {}).get("generation", {})
    enabled = set(generation.get("enabled_strategies", registry.names()))
    unknown = set(quotas) - set(registry.names())
    if unknown:
        raise ValueError(f"未知策略配额：{sorted(unknown)}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in quotas.values()):
        raise ValueError("配额必须是非负整数")
    disabled = {name for name, quota in quotas.items() if quota and name not in enabled}
    if disabled:
        raise ValueError(f"策略已禁用但配额非零：{sorted(disabled)}")
    context = {
        "manager": manager,
        "phase_references": phase_references,
        "frameworks": (framework_enumerator or enumerate_legal_frameworks)(manager.boundary, phase_references),
        "parents": parent_branch_ids or sorted(manager.data["branches"]),
    }
    common = {"site_mappings": site_mappings, "oxidation_states": oxidation_states}
    offsets = generation.get("strategy_seed_offsets", {})
    candidates = []
    for index, (name, quota) in enumerate(quotas.items()):
        if not quota:
            continue
        options = {**common, **(strategy_options or {}).get(name, {})}
        produced = registry.get(name)(context, quota, seed + int(offsets.get(name, index * 10000)), options)
        for candidate in produced:
            candidate.setdefault("strategy", name)
        candidates.extend(produced)
    for candidate in candidates:
        before = set(manager.data["branches"])
        candidate["branch_id"] = manager.add_branch(P=candidate["P"], H=candidate["H"], x=candidate["x"], T=candidate["T"], composition=candidate.get("composition")) if register else None
        candidate["already_registered"] = candidate["branch_id"] in before
    return candidates
