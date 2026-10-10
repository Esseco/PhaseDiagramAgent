"""计算 MLIP 委员会的能量和力预测分歧。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def evaluate_qbc(
    structure: Any,
    committee: dict[str, Any],
    *,
    predictor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """返回委员会分歧；分歧是采样信号，不是真实预测误差。"""
    members = committee.get("loaded_members", [])
    if predictor is None:
        return _missing("QBC predictor 未配置")
    if len(members) < 2:
        return _missing("至少需要两个内容不同且已加载的模型")
    predictions = []
    try:
        for member in members:
            result = predictor(structure=structure, model=member["model"], member=member)
            energy = float(result["energy"])
            forces = np.asarray(result["forces"], dtype=float)
            if forces.shape != (len(structure), 3):
                raise ValueError(f"模型 {member['model_id']} 的 forces shape={forces.shape}")
            predictions.append(
                {
                    "model_id": member["model_id"],
                    "energy": energy,
                    "forces": forces,
                    "force_rms": float(np.sqrt(np.mean(forces**2))),
                    "force_max": float(np.linalg.norm(forces, axis=1).max(initial=0)),
                }
            )
        energies = np.asarray([item["energy"] for item in predictions])
        forces = np.stack([item["forces"] for item in predictions])
        force_std = np.std(forces, axis=0)
        atom_disagreement = np.linalg.norm(force_std, axis=1)
        force_values = atom_disagreement if len(atom_disagreement) else np.asarray([0.0])
        return {
            "status": "completed",
            "member_count": len(predictions),
            "energy_mean": float(energies.mean()),
            "energy_std": float(energies.std()),
            "energy_range": float(np.ptp(energies)),
            "energy_per_atom_std": float(energies.std() / len(structure)),
            "force_rms_disagreement": float(np.sqrt(np.mean(force_std**2))),
            "force_max_atom_disagreement": float(atom_disagreement.max(initial=0)),
            "f_std_max": float(atom_disagreement.max(initial=0)),
            "f_std_p95": float(np.percentile(force_values, 95)),
            "predictions": [
                {
                    "model_id": item["model_id"],
                    "energy": item["energy"],
                    "energy_per_atom": item["energy"] / len(structure),
                    "force_rms": item["force_rms"],
                    "force_max": item["force_max"],
                }
                for item in predictions
            ],
            "interpretation": "committee_disagreement_not_true_error",
            "error": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "interpretation": "committee_disagreement_not_true_error",
        }


def _missing(message):
    return {
        "status": "not_configured",
        "error": message,
        "interpretation": "committee_disagreement_not_true_error",
    }
