"""保持 P、H、T，仅改变可实现的 x。"""

from __future__ import annotations

import random
from typing import Any

from scientific_layer.structures.identify_branch import identify_branch_parameters
from scientific_layer.structures.generate_branch_structure import generate_branch_structure


def generate_composition_branches(
    manager: Any,
    frameworks: list[dict[str, Any]],
    phase_references: dict[str, Any],
    *,
    parent_branch_ids: list[str],
    quota: int,
    seed: int,
    oxidation_states: dict[str, int | float] | None = None,
    preserve_reference_tm: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks = []
    for parent_id in parent_branch_ids:
        parent = _branch(manager, parent_id)
        framework = _framework(frameworks, parent["P"], parent["H"])
        for x in framework["allowed_x"]:
            if x != parent["x"]:
                tasks.append((parent_id, parent, x))
    rng.shuffle(tasks)

    candidates = []
    for offset, (parent_id, parent, x) in enumerate(tasks[:quota]):
        current_seed = seed + offset
        structure = generate_branch_structure(
            manager.boundary,
            phase=parent["P"],
            H=parent["H"],
            x=x,
            T=None if preserve_reference_tm else parent["T"],
            preserve_reference_tm=preserve_reference_tm,
            phase_references=phase_references,
            oxidation_states=oxidation_states,
            seed=current_seed,
        )
        parameters = identify_branch_parameters(
            structure, manager.boundary, phase_references=phase_references,
            phase_hint=parent["P"],
        )
        candidates.append(
            _candidate(parameters, structure, "composition", parent_id, current_seed)
        )
    return candidates


def _branch(manager: Any, branch_id: str) -> dict[str, Any]:
    try:
        return manager.data["branches"][branch_id]
    except KeyError as error:
        raise KeyError(f"未知父 branch：{branch_id}") from error


def _framework(frameworks: list[dict[str, Any]], P: str, H: Any) -> dict[str, Any]:
    matches = [item for item in frameworks if item["P"] == P and item["H"] == H]
    if len(matches) != 1:
        raise ValueError(f"无法为 P={P}, H={H} 唯一确定框架")
    return matches[0]


def _candidate(
    parameters: dict[str, Any], structure: Any, strategy: str, parent: str, seed: int
) -> dict[str, Any]:
    return {
        **{
            key: parameters[key] for key in ("P", "H", "det_H", "x", "T", "composition")
        },
        "structure": structure,
        "strategy": strategy,
        "parent_branch_id": parent,
        "seed": seed,
        "inheritance": "preserved_T",
    }
