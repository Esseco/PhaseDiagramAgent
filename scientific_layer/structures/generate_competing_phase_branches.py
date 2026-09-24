"""在相同 x 下生成其他允许相的 branch。"""

from __future__ import annotations

import random
from typing import Any

from scientific_layer.structures.identify_branch import identify_branch_parameters
from scientific_layer.structures.generate_branch_structure import generate_branch_structure


def generate_competing_phase_branches(
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
            if target["P"] != parent["P"] and parent["x"] in target["allowed_x"]:
                tasks.append((parent_id, parent, source_key, target))
    random.Random(seed).shuffle(tasks)

    candidates = []
    for offset, (parent_id, parent, source_key, target) in enumerate(tasks[:quota]):
        current_seed = seed + offset
        target_key = _framework_key(target["P"], target["H"])
        T, inheritance = _inherit_T(parent["T"], mappings.get((source_key, target_key)))
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
            _candidate(parameters, structure, parent_id, current_seed, inheritance)
        )
    return candidates


def _inherit_T(T: list[str], mapping: list[int] | None) -> tuple[list[str] | None, str]:
    if mapping is None:
        return None, "regenerated_no_mapping"
    if any(index < 0 or index >= len(T) for index in mapping):
        raise ValueError("跨相位点映射含越界索引")
    return [T[index] for index in mapping], "explicit_site_mapping"


def _framework_key(P: str, H: Any) -> str:
    return f"{P}:{H}"


def _branch(manager: Any, branch_id: str) -> dict[str, Any]:
    try:
        return manager.data["branches"][branch_id]
    except KeyError as error:
        raise KeyError(f"未知父 branch：{branch_id}") from error


def _candidate(
    parameters: dict[str, Any],
    structure: Any,
    parent: str,
    seed: int,
    inheritance: str,
) -> dict[str, Any]:
    return {
        **{
            key: parameters[key] for key in ("P", "H", "det_H", "x", "T", "composition")
        },
        "structure": structure,
        "strategy": "competing_phase",
        "parent_branch_id": parent,
        "seed": seed,
        "inheritance": inheritance,
    }
