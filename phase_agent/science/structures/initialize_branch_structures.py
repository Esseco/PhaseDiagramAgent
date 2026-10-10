"""为 branch 生成多个 Na/空位初态，只改变占位向量 V。"""

from __future__ import annotations

import copy
import hashlib
import random
from fractions import Fraction
from typing import Any

from phase_agent.science.structures.boundary_utils import compact_json, normalize_fraction
from phase_agent.science.structures.identify_branch import extract_T
from phase_agent.science.structures.generate_branch_structure import generate_branch_structure
from phase_agent.science.structures.rank_na_orderings_by_electrostatics import (
    rank_na_orderings_by_electrostatics,
)


def initialize_branch_structures(
    branches: list[dict[str, Any]],
    boundary: dict[str, Any],
    phase_references: dict[str, Any],
    *,
    initial_states_per_branch: int,
    seed: int,
    preserve_reference_tm: bool = False,
) -> list[dict[str, Any]]:
    """固定 P、H、x、T，为每个 branch 生成至多指定数量的不同 V。"""
    if (
        isinstance(initial_states_per_branch, bool)
        or not isinstance(initial_states_per_branch, int)
        or initial_states_per_branch <= 0
    ):
        raise ValueError("initial_states_per_branch 必须是正整数")

    output = []
    for branch_index, original in enumerate(branches):
        missing = {"P", "H", "x", "T"} - original.keys()
        if missing:
            raise ValueError(f"branch 缺少字段：{sorted(missing)}")
        branch_seed = seed + branch_index
        target_x = Fraction(normalize_fraction(original["x"]))
        full = generate_branch_structure(
            boundary,
            phase=original["P"],
            H=original["H"],
            x=0 if target_x == 0 else 1,
            T=None if preserve_reference_tm else list(original["T"]),
            preserve_reference_tm=preserve_reference_tm,
            phase_references=phase_references,
            seed=branch_seed,
            # Internal full-Na template only.  The final structures below are
            # restored to original["x"] and retain the branch boundary rule.
            enforce_phase_composition=False,
        )
        na_indices = [
            index
            for index, site in enumerate(full)
            if site.is_ordered and site.specie.symbol == "Na"
        ]
        oxygen_count = sum(site.is_ordered and site.specie.symbol == "O" for site in full)
        occupied = target_x * oxygen_count / 2
        if occupied.denominator != 1 or not 0 <= occupied <= len(na_indices):
            raise ValueError(f"x={original['x']} 无法在 {len(na_indices)} 个 Na 位点上实现")
        try:
            ranked = rank_na_orderings_by_electrostatics(full, original["x"], limit=10)
        except Exception as error:
            raise RuntimeError(
                f"branch {original.get('branch_id', original.get('candidate_id', branch_index))} "
                f"静电能初态生成失败：{error}"
            ) from error
        count = min(initial_states_per_branch, 3, len(ranked))
        selected_ranks = random.Random(branch_seed).sample(range(len(ranked)), count)
        for state_index, valid_rank in enumerate(selected_ranks):
            structure, electrostatic_energy, charge_scheme, raw_rank = ranked[valid_rank]
            chosen = _na_occupancy_indices(full, structure, na_indices)
            chosen_set = set(chosen)
            structure.sort()
            actual_T, _ = extract_T(structure, boundary["TM_ratio"])
            if actual_T != list(original["T"]):
                raise RuntimeError("初始化 Na/空位后 T 发生变化")
            V = [int(index in chosen_set) for index in range(len(na_indices))]
            candidate = copy.copy(original)
            candidate.update(
                {
                    "candidate_id": _candidate_id(original, V),
                    "structure": structure,
                    "full_na_structure": full.copy() if target_x > 0 else None,
                    "V": V,
                    "arrangement": {
                        "V": V,
                        "occupied_sites": list(chosen),
                        "vacancy_sites": [i for i, value in enumerate(V) if not value],
                    },
                    "initial_state_index": state_index,
                    "initial_state_count": count,
                    "initialization_seed": branch_seed,
                    "initialization_method": "electrostatic_top10_random3_layer_occupied",
                    "electrostatic_rank": valid_rank,
                    "electrostatic_raw_rank": raw_rank,
                    "electrostatic_rank_pool_size": len(ranked),
                    "electrostatic_energy": electrostatic_energy,
                    "electrostatic_charge_scheme": charge_scheme,
                }
            )
            output.append(candidate)
    return output


def _na_occupancy_indices(full, ordered, na_indices):
    import numpy as np

    positions = [
        site.frac_coords for site in ordered if site.is_ordered and site.specie.symbol == "Na"
    ]
    chosen = []
    for position, index in enumerate(na_indices):
        reference = full[index].frac_coords
        if any(
            np.allclose((coords - reference) - np.rint(coords - reference), 0, atol=1e-5)
            for coords in positions
        ):
            chosen.append(position)
    if len(chosen) != len(positions):
        raise RuntimeError("静电能构型的 Na 位点无法映射回母结构")
    return tuple(chosen)


def _candidate_id(branch, V):
    payload = [
        branch["P"],
        branch["H"],
        normalize_fraction(branch["x"]),
        branch["T"],
        V,
    ]
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"V-{digest}"
