"""为 branch 生成多个 Na/空位初态，只改变占位向量 V。"""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import random
from fractions import Fraction
from typing import Any

from scientific_layer.structures.boundary_utils import compact_json, normalize_fraction
from scientific_layer.structures.identify_branch import extract_T
from scientific_layer.structures.generate_branch_structure import generate_branch_structure


def initialize_branch_structures(
    branches: list[dict[str, Any]],
    boundary: dict[str, Any],
    phase_references: dict[str, Any],
    *,
    initial_states_per_branch: int,
    seed: int,
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
        full = generate_branch_structure(
            boundary,
            phase=original["P"],
            H=original["H"],
            x=1,
            T=list(original["T"]),
            phase_references=phase_references,
            seed=branch_seed,
        )
        na_indices = [
            index
            for index, site in enumerate(full)
            if site.is_ordered and site.specie.symbol == "Na"
        ]
        oxygen_count = sum(
            site.is_ordered and site.specie.symbol == "O" for site in full
        )
        occupied = Fraction(normalize_fraction(original["x"])) * oxygen_count / 2
        if occupied.denominator != 1 or not 0 <= occupied <= len(na_indices):
            raise ValueError(
                f"x={original['x']} 无法在 {len(na_indices)} 个 Na 位点上实现"
            )
        choices = _choose_occupancies(
            len(na_indices),
            occupied.numerator,
            initial_states_per_branch,
            branch_seed,
        )
        for state_index, chosen in enumerate(choices):
            chosen_set = set(chosen)
            structure = full.copy()
            remove = [
                site_index
                for position, site_index in enumerate(na_indices)
                if position not in chosen_set
            ]
            structure.remove_sites(remove)
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
                    "V": V,
                    "arrangement": {
                        "V": V,
                        "occupied_sites": list(chosen),
                        "vacancy_sites": [i for i, value in enumerate(V) if not value],
                    },
                    "initial_state_index": state_index,
                    "initial_state_count": len(choices),
                    "initialization_seed": branch_seed,
                }
            )
            output.append(candidate)
    return output


def _choose_occupancies(site_count, occupied_count, requested, seed):
    total = math.comb(site_count, occupied_count)
    count = min(requested, total)
    rng = random.Random(seed)
    if total <= 100_000:
        choices = list(itertools.combinations(range(site_count), occupied_count))
        rng.shuffle(choices)
        return choices[:count]
    choices = set()
    attempts = max(100, count * 100)
    for _ in range(attempts):
        choices.add(tuple(sorted(rng.sample(range(site_count), occupied_count))))
        if len(choices) == count:
            break
    if len(choices) < count:
        raise RuntimeError(f"只生成 {len(choices)}/{count} 个不同 V")
    return sorted(choices)


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
