"""用原子数、初态数量和计划预算估计相对成本。"""

from __future__ import annotations

import copy
from typing import Any

from phase_agent.science.features.calculate_size_aware_cost import calculate_size_aware_cost


def estimate_candidate_cost(
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """返回带 ``estimated_cost`` 的候选副本；不估计真实耗时。

    默认公式：``(atom_count/reference_atoms)^atom_exponent ×
    initial_state_count × planned_search_budget × scale``。
    """
    settings = {
        "reference_atoms": 1.0,
        "atom_exponent": 1.0,
        "scale": 1.0,
        "planned_search_budget": 1.0,
    }
    settings.update(config or {})
    for key in settings:
        if not isinstance(settings[key], (int, float)) or isinstance(settings[key], bool):
            raise TypeError(f"成本参数 {key} 必须是数值")
    if settings["reference_atoms"] <= 0 or settings["atom_exponent"] < 0:
        raise ValueError("reference_atoms 必须为正数，atom_exponent 不能为负")
    if settings["scale"] < 0 or settings["planned_search_budget"] < 0:
        raise ValueError("scale 和 planned_search_budget 不能为负")

    output = []
    for original in candidates:
        candidate = copy.copy(original)
        atom_count = _atom_count(candidate)
        initial_count = candidate.get("initial_state_count", 1)
        budget = candidate.get("planned_search_budget", settings["planned_search_budget"])
        _positive(initial_count, "initial_state_count")
        _nonnegative(budget, "planned_search_budget")
        record = calculate_size_aware_cost(
            atom_count=atom_count,
            evaluation_count=budget,
            initial_state_count=initial_count,
            config=settings,
        )
        candidate["estimated_cost"] = {
            "kind": "relative_cost_units",
            "value": record["value"],
            "atom_count": atom_count,
            "initial_state_count": initial_count,
            "planned_search_budget": float(budget),
            "model": dict(settings),
            "wall_time": None,
        }
        output.append(candidate)
    return output


def _atom_count(candidate):
    if candidate.get("structure") is not None:
        value = len(candidate["structure"])
    else:
        value = candidate.get("atom_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("候选必须提供结构或正整数 atom_count")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} 必须是非负数")
