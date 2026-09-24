"""用通过验证的新模型重评估关键候选并生成重算标记。"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np


def reevaluate_candidates(
    candidates: list[dict[str, Any]],
    old_model: dict[str, Any],
    new_model: dict[str, Any],
    validation_result: dict[str, Any],
    *,
    predictor: Callable[..., dict[str, Any]] | None = None,
    thresholds: dict[str, float] | None = None,
    manager: Any | None = None,
) -> dict[str, Any]:
    """只有新模型验证通过才重评估；本函数不直接提交重算任务。"""
    if not validation_result.get("passed"):
        return {
            "status": "blocked",
            "candidates": [],
            "reason": "new_model_not_validated",
        }
    if predictor is None:
        return {
            "status": "not_configured",
            "candidates": [],
            "reason": "predictor_not_configured",
        }
    limits = {
        "energy_change_for_relax": 0.05,
        "force_change_for_relax": 0.2,
        "energy_change_for_search": 0.1,
    }
    limits.update(thresholds or {})
    output = []
    for original in candidates:
        item = copy.copy(original)
        old = predictor(structure=item["structure"], model=old_model)
        new = predictor(structure=item["structure"], model=new_model)
        energy_change = abs(float(new["energy"]) - float(old["energy"])) / len(
            item["structure"]
        )
        old_forces, new_forces = np.asarray(old["forces"]), np.asarray(new["forces"])
        force_change = float(
            np.max(np.linalg.norm(new_forces - old_forces, axis=1), initial=0)
        )
        item["reevaluation"] = {
            "old_model_version": old_model.get("version"),
            "new_model_version": new_model.get("version"),
            "energy_change_per_atom": energy_change,
            "max_force_change": force_change,
            "needs_relax": energy_change >= limits["energy_change_for_relax"]
            or force_change >= limits["force_change_for_relax"],
            "needs_search": energy_change >= limits["energy_change_for_search"],
            "status": "pending",
        }
        structure_id = item.get("structure_id")
        if manager is not None and structure_id in manager.data.get("structures", {}):
            record = manager.data["structures"][structure_id]
            metadata = record.get("metadata") or {}
            record["metadata"] = metadata
            metadata["reevaluation"] = copy.deepcopy(item["reevaluation"])
        output.append(item)
    return {"status": "completed", "candidates": output, "thresholds": limits}
