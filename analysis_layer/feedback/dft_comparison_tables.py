"""Build same-frame parity tables and their MAE/RMSE from saved predictions."""
import json
import math

import numpy as np


IDENTITY_FIELDS = (
    "epoch",
    "model_version", "model_sha256", "search_group_index", "dft_round", "upload_operation_id",
    "parent_relax_round", "task_id", "structure_id", "branch_id", "stage",
    "frame_index", "phase", "x_Na_per_O2", "composition", "atom_count",
    "spin_check_status", "comparison_status", "reason",
)
ENERGY_FIELDS = (*IDENTITY_FIELDS, "dft_energy_eV", "mlip_energy_eV",
                 "energy_error_eV", "energy_abs_error_eV", "dft_energy_eV_per_atom",
                 "mlip_energy_eV_per_atom", "energy_error_eV_per_atom",
                 "energy_abs_error_eV_per_atom")
FORCE_FIELDS = (*IDENTITY_FIELDS, "atom_index", "element", "component",
                "dft_force_eV_per_A", "mlip_force_eV_per_A", "force_error_eV_per_A",
                "force_abs_error_eV_per_A")
METRIC_FIELDS = (
    "epoch",
    "model_version", "search_group_index", "dft_round", "upload_operation_id",
    "parent_relax_round", "metric", "mae", "rmse", "unit", "sample_count",
    "recovered_tasks", "expected_tasks", "pending_tasks", "matched_structures",
    "not_evaluated_structures", "status", "averaging",
)


def build_comparison_tables(records, comparisons, *, round_name):
    """One energy row per task, one force row per atom/component; no inference."""
    by_task = {row["task_id"]: row for row in comparisons}
    energies, forces = [], []
    for record in sorted(records, key=lambda row: str(row["task_id"])):
        comparison = by_task.get(record["task_id"], {})
        predicted, reason = _verified_prediction(record, comparison)
        composition = record.get("composition") or {}
        oxygen = composition.get("O") or 0
        n = int(record.get("atom_count") or 0)
        identity = {key: record.get(key) for key in IDENTITY_FIELDS}
        identity.update(
            spin_check_status=(record.get("spin_state_check") or {}).get("status"),
            model_sha256=(record.get("mlip_prediction") or {}).get("model_sha256"),
            dft_round=round_name, frame_index=record.get("training_frame_index"),
            phase=record.get("actual_phase"),
            x_Na_per_O2=format(2 * float(composition.get("Na", 0)) / float(oxygen), ".10f") if oxygen else None,
            composition=json.dumps(composition, ensure_ascii=False, sort_keys=True),
            comparison_status="completed" if predicted is not None else "not_evaluated", reason=reason,
        )
        dft_energy = _finite(record.get("energy")) if record.get("energy_unit") == "eV" else None
        mlip_energy = float(predicted["energy"]) if predicted is not None else None
        error = mlip_energy - dft_energy if mlip_energy is not None else None
        energies.append({**identity, "dft_energy_eV": dft_energy, "mlip_energy_eV": mlip_energy,
                         "energy_error_eV": error, "energy_abs_error_eV": abs(error) if error is not None else None,
                         "dft_energy_eV_per_atom": dft_energy/n if dft_energy is not None and n > 0 else None,
                         "mlip_energy_eV_per_atom": mlip_energy/n if mlip_energy is not None else None,
                         "energy_error_eV_per_atom": error/n if error is not None else None,
                         "energy_abs_error_eV_per_atom": abs(error)/n if error is not None else None})
        dft_forces = _valid_forces(record, n)
        if dft_forces is None:
            continue
        sites = (record.get("structure") or {}).get("sites") or []
        for atom_index, vector in enumerate(dft_forces):
            species = sites[atom_index].get("species") if atom_index < len(sites) else None
            # Do not reorder atoms, infer species from composition or invent labels.
            element = species[0].get("element") if species and len(species) == 1 else None
            for axis, component in enumerate("xyz"):
                dft_force = float(vector[axis])
                mlip_force = float(predicted["forces"][atom_index][axis]) if predicted is not None else None
                error = mlip_force - dft_force if mlip_force is not None else None
                forces.append({**identity, "atom_index": atom_index, "element": element,
                               "component": component, "dft_force_eV_per_A": dft_force,
                               "mlip_force_eV_per_A": mlip_force, "force_error_eV_per_A": error,
                               "force_abs_error_eV_per_A": abs(error) if error is not None else None})
    return energies, forces


def comparison_metrics(energies, forces, scope):
    """JSON and CSV statistics share the same fully paired rows and weights."""
    paired = [row for row in energies if row["comparison_status"] == "completed"]
    metrics = {"round_scope": scope, "matched_structures": len(paired),
               "evaluation_type": "original_round_model_same_frame",
               "interpretation": "原轮次冻结模型同帧DFT比较；不混同训练误差或K折误差。独立测试还需确认结构未参与该模型训练。",
               "not_evaluated": [{"task_id": row["task_id"], "round_scope": scope,
                                  "status": row["comparison_status"], "reason": row["reason"]}
                                 for row in energies if row["comparison_status"] != "completed"]}
    for name, rows, field, unit in (
        ("energy_total", paired, "energy_error_eV", "eV"),
        ("energy_per_atom", paired, "energy_error_eV_per_atom", "eV/atom"),
        ("forces", [row for row in forces if row["comparison_status"] == "completed"], "force_error_eV_per_A", "eV/angstrom"),
    ):
        values = np.asarray([row[field] for row in rows], dtype=float)
        metrics[name] = {"mae": float(np.abs(values).mean()) if len(values) else None,
                         "rmse": float(np.sqrt((values**2).mean())) if len(values) else None,
                         "mae_unit": unit, "rmse_unit": unit}
        if name == "forces":
            metrics[name].update(components=len(values), averaging="atomic Cartesian components")
    return metrics


def _verified_prediction(record, comparison):
    from scientific_layer.dft.spin_acceptance import spin_standard_passed
    if not spin_standard_passed(record):
        return None, "DFT Fe/Mn spin standard not passed"
    if comparison.get("status") != "completed":
        return None, comparison.get("reason") or "same-frame MLIP comparison unavailable"
    try:
        if record.get("status") != "completed" or record.get("converged") is not True or record.get("checks_passed") is False:
            raise ValueError("DFT final frame not completed/converged/verified")
        if record.get("training_ready") is not True:
            raise ValueError("verified portable DFT labels unavailable")
        if record.get("final_frame_valid") is False:
            raise ValueError("actual DFT final frame invalid")
        if record.get("final_frame_index") is not None and record["final_frame_index"] != record.get("training_frame_index"):
            raise ValueError("training frame is not the actual DFT final frame")
        predicted = record.get("mlip_prediction") or {}
        if not record.get("model_version") or predicted.get("model_version") != record["model_version"]:
            raise ValueError("matching-version saved MLIP prediction unavailable")
        if record.get("energy_unit") != "eV" or predicted.get("energy_unit") != "eV":
            raise ValueError("energy unit must be eV")
        if _finite(record.get("energy")) is None or _finite(predicted.get("energy")) is None:
            raise ValueError("finite same-frame energies unavailable")
        if _finite(float(predicted["energy"]) - float(record["energy"])) is None:
            raise ValueError("finite same-frame energy difference unavailable")
        n = int(record.get("atom_count") or 0)
        if _valid_forces(record, n) is None or _valid_forces(predicted, n) is None:
            raise ValueError("finite Nx3 forces in eV/angstrom unavailable")
        return predicted, None
    except (ValueError, TypeError, KeyError) as error:
        return None, str(error)


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _valid_forces(record, n):
    if n <= 0 or record.get("forces_unit") != "eV/angstrom":
        return None
    try:
        values = np.asarray(record.get("forces"), dtype=float)
        return values if values.shape == (n, 3) and np.isfinite(values).all() else None
    except (TypeError, ValueError):
        return None
