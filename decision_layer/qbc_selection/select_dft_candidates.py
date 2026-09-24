"""结合凸包影响、QBC、多样性和独立抽查选择 DFT 候选。"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np


def select_dft_candidates(
    candidates: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    weights: dict[str, float] | None = None,
    audit_fraction: float = 0.1,
    cost_budget: float | None = None,
) -> dict[str, Any]:
    """加权选择并保留少量不依赖评分的随机抽查。"""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size 必须是正整数")
    if not 0 <= audit_fraction <= 1:
        raise ValueError("audit_fraction 必须在 [0,1]")
    if cost_budget is not None and cost_budget < 0:
        raise ValueError("cost_budget 不能为负")
    config = {"hull_impact": 1.0, "qbc": 1.0, "diversity": 0.5}
    config.update(weights or {})
    pool = [copy.copy(item) for item in candidates]
    rng = random.Random(seed)
    rng.shuffle(pool)
    audit_count = min(len(pool), round(batch_size * audit_fraction))
    audit, remaining = pool[:audit_count], pool[audit_count:]
    hull = _normalized([item.get("hull_impact", 0.0) for item in remaining])
    qbc_values = [_qbc_value(item) for item in remaining]
    qbc = _normalized_known(qbc_values)
    diversity = _diversity(remaining)
    for item, h_value, q_value, d_value in zip(
        remaining, hull, qbc, diversity, strict=True
    ):
        components = [(config["hull_impact"], h_value), (config["diversity"], d_value)]
        if not np.isnan(q_value):
            components.append((config["qbc"], q_value))
        total_weight = sum(weight for weight, _ in components)
        item["dft_selection_score"] = sum(weight * value for weight, value in components) / total_weight
        item["dft_selection_mode"] = "ranked"
        item["qbc_evidence_status"] = "known" if _qbc_value(item) is not None else "unknown"
    remaining.sort(
        key=lambda item: (
            -item["dft_selection_score"],
            str(item.get("structure_id", item.get("candidate_id", ""))),
        )
    )
    for item in audit:
        item["dft_selection_mode"] = "independent_audit"
        item["dft_selection_score"] = None
    selected, rejected, total_cost = [], [], 0.0
    for item in [*audit, *remaining]:
        cost = _cost(item)
        if len(selected) >= batch_size:
            rejected.append({"candidate": item, "reason": "batch_size"})
        elif cost_budget is not None and total_cost + cost > cost_budget:
            rejected.append({"candidate": item, "reason": "cost_budget"})
        else:
            selected.append(item)
            total_cost += cost
    return {
        "selected_candidates": selected,
        "rejected_candidates": rejected,
        "summary": {
            "selected": len(selected),
            "audit_selected": sum(
                item["dft_selection_mode"] == "independent_audit" for item in selected
            ),
            "relative_cost": total_cost,
            "seed": seed,
            "weights": config,
        },
    }


def _normalized(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or np.ptp(array) == 0:
        return np.zeros(len(array))
    return (array - array.min()) / np.ptp(array)


def _normalized_known(values):
    """Normalize known observations without representing missing evidence as zero."""
    known = [float(value) for value in values if value is not None]
    if not known:
        return np.full(len(values), np.nan)
    low, high = min(known), max(known)
    return np.asarray([np.nan if value is None else (0.5 if high == low else
        (float(value) - low) / (high - low)) for value in values], dtype=float)


def _qbc_value(item):
    qbc = item.get("qbc", {})
    return qbc.get("force_max_atom_disagreement", qbc.get("energy_per_atom_std"))


def _diversity(items):
    vectors = [item.get("features") for item in items]
    if not vectors or any(value is None for value in vectors):
        return np.asarray([item.get("diversity_score", 0.0) for item in items])
    matrix = np.asarray(vectors, dtype=float)
    center = matrix.mean(axis=0)
    return _normalized(np.linalg.norm(matrix - center, axis=1))


def _cost(item):
    record = item.get("estimated_cost", 0.0)
    return float(record.get("value", 0.0) if isinstance(record, dict) else record)
