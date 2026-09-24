"""在同相中扩胞或改变超胞形状。"""

from __future__ import annotations

import random
from typing import Any

from scientific_layer.structures.identify_branch import identify_branch_parameters
from scientific_layer.structures.generate_branch_structure import generate_branch_structure


def generate_periodic_extension_branches(
    manager: Any,
    frameworks: list[dict[str, Any]],
    phase_references: dict[str, Any],
    *,
    parent_branch_ids: list[str],
    quota: int,
    seed: int,
    site_mappings: dict[tuple[str, str], list[int]] | None = None,
    oxidation_states: dict[str, int | float] | None = None,
) -> list[dict[str, Any]]:
    mappings = site_mappings or {}
    tasks = []
    for parent_id in parent_branch_ids:
        parent = _branch(manager, parent_id)
        source_key = _framework_key(parent["P"], parent["H"])
        for target in frameworks:
            if (
                target["P"] == parent["P"]
                and target["H"] != parent["H"]
                and parent["x"] in target["allowed_x"]
            ):
                tasks.append((parent_id, parent, source_key, target))
    random.Random(seed).shuffle(tasks)

    candidates = []
    for offset, (parent_id, parent, source_key, target) in enumerate(tasks[:quota]):
        current_seed = seed + offset
        target_key = _framework_key(target["P"], target["H"])
        mapping = mappings.get((source_key, target_key))
        if mapping is None:
            T, inheritance = None, "regenerated_no_mapping"
        else:
            if any(index < 0 or index >= len(parent["T"]) for index in mapping):
                raise ValueError("跨超胞位点映射含越界索引")
            T = [parent["T"][index] for index in mapping]
            inheritance = "explicit_site_mapping"
        structure = generate_branch_structure(
            manager.boundary,
            phase=target["P"],
            H=target["H"],
            x=parent["x"],
            T=T,
            phase_references=phase_references,
            oxidation_states=oxidation_states,
            seed=current_seed,
        )
        parameters = identify_branch_parameters(
            structure, manager.boundary, phase_references=phase_references
        )
        candidates.append(
            {
                **{
                    name: parameters[name]
                    for name in ("P", "H", "det_H", "x", "T", "composition")
                },
                "structure": structure,
                "strategy": "periodic_extension",
                "parent_branch_id": parent_id,
                "seed": current_seed,
                "inheritance": inheritance,
                "changes": {"H": {"from": parent["H"], "to": target["H"]}},
            }
        )
    return candidates


def _framework_key(P: str, H: Any) -> str:
    return f"{P}:{H}"


def _branch(manager: Any, branch_id: str) -> dict[str, Any]:
    try:
        return manager.data["branches"][branch_id]
    except KeyError as error:
        raise KeyError(f"未知父 branch：{branch_id}") from error
