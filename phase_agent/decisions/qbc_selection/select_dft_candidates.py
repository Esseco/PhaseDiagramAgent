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
    cover_phases: bool = False,
) -> dict[str, Any]:
    """加权选择并保留少量不依赖评分的随机抽查。"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
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
    for item, h_value, q_value, d_value in zip(remaining, hull, qbc, diversity, strict=True):
        components = [(config["hull_impact"], h_value), (config["diversity"], d_value)]
        if not np.isnan(q_value):
            components.append((config["qbc"], q_value))
        total_weight = sum(weight for weight, _ in components)
        item["dft_selection_score"] = (
            sum(weight * value for weight, value in components) / total_weight
        )
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
    coverage = []
    if cover_phases:
        phases = sorted({item.get("phase") for item in pool if item.get("phase")})
        for phase in phases:
            options = [item for item in remaining if item.get("phase") == phase]
            if not options:
                options = [item for item in audit if item.get("phase") == phase]
            if options:
                coverage.append(options[0])
        coverage_ids = {id(item) for item in coverage}
        audit = [item for item in audit if id(item) not in coverage_ids]
        remaining = [item for item in remaining if id(item) not in coverage_ids]
        for item in coverage:
            item["dft_selection_mode"] = "phase_coverage"
    selected, rejected, total_cost = [], [], 0.0
    if cover_phases:
        # Spread Na compositions after phase representatives, while retaining
        # hull/QBC scores rather than replacing the scientific ranking.
        ordered = list(coverage) + list(audit)
        ranked = list(remaining)
        while ranked:

            def spread_score(item):
                x = item.get("x_Na_per_O2")
                known = [
                    row.get("x_Na_per_O2")
                    for row in ordered
                    if isinstance(row.get("x_Na_per_O2"), (int, float))
                ]
                spread = (
                    min(abs(float(x) - float(value)) for value in known)
                    if known and isinstance(x, (int, float))
                    else 0.0
                )
                return float(item.get("dft_selection_score") or 0.0) + config["diversity"] * spread

            best = max(ranked, key=spread_score)
            ordered.append(best)
            ranked.remove(best)
    else:
        ordered = [*audit, *remaining]
    for item in ordered:
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
    return np.asarray(
        [
            np.nan
            if value is None
            else (0.5 if high == low else (float(value) - low) / (high - low))
            for value in values
        ],
        dtype=float,
    )


def _qbc_value(item):
    qbc = item.get("qbc") or {}
    if str(qbc.get("status", "")).startswith(("not_", "failed")):
        return None
    value = qbc.get(
        "f_std_max", qbc.get("force_max_atom_disagreement", qbc.get("energy_per_atom_std"))
    )
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


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
