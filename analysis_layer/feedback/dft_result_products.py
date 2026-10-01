"""Training records and round-scoped paired DFT/MLIP error products."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def record_dft_products(state, result, evaluator=None, manager=None):
    if result.get("stage") not in {"dft_single_point", "dft_relax"} or result.get("status") != "completed":
        return
    outputs = result.get("outputs") or {}
    task = next((t for t in state.get("tasks", []) if t.get("task_id") == result.get("task_id")), {})
    lineage = {key: result.get(key, task.get(key)) for key in (
        "model_version", "upload_operation_id", "search_group_index", "parent_relax_round", "batch_id")}
    # upload_operation_id identifies the DFT generation round; do not aggregate
    # separate manual batches into a falsely named round when it is absent.
    scope = {key: value for key, value in lineage.items() if key != "batch_id"}
    if not scope.get("upload_operation_id"):
        scope["batch_id"] = lineage.get("batch_id") or result.get("task_id")
    stored_structure = ((manager.data.get("structures") or {}).get(result.get("structure_id")) or {}) if manager is not None else {}
    branch_id = result.get("branch_id") or task.get("branch_id") or stored_structure.get("branch_id")
    record = {**lineage, "task_id": result.get("task_id"),
              "structure_id": result.get("structure_id"), "stage": result.get("stage"),
              "branch_id": branch_id,
              "status": "completed", "converged": result.get("converged"),
              "checks_passed": result.get("checks_passed", True),
              "energy": outputs.get("training_energy", outputs.get("energy")),
              "energy_unit": outputs.get("energy_unit"), "round_scope": scope,
              **{key: deepcopy(outputs[key]) for key in (
                  "structure", "composition", "atom_count", "forces", "forces_unit",
                  "stress", "stress_unit", "stress_convention", "training_ready",
                  "training_schema", "training_frame_index", "training_energy_kind",
                  "actual_phase", "phase_identification", "structure_path") if key in outputs}}
    state.setdefault("dft_dataset_records", []).append(record)
    if outputs.get("training_ready") is True:
        if record["task_id"] not in {r.get("task_id") for r in state.setdefault("new_dft_records", [])}:
            state["new_dft_records"].append(deepcopy(record))
    comparison = {"task_id": record["task_id"], "round_scope": scope, "status": "not_evaluated"}
    try:
        if outputs.get("training_ready") is not True:
            raise ValueError(outputs.get("training_error") or "portable DFT labels unavailable")
        if not callable(evaluator):
            raise ValueError("same-frame MLIP evaluator unavailable")
        predicted = evaluator(result=deepcopy(result), structure_id=record["structure_id"], manager=manager)
        if not isinstance(predicted, dict) or predicted.get("model_version") != record["model_version"]:
            raise ValueError("comparison requires matching explicit model_version")
        if predicted.get("energy_unit") != "eV" or predicted.get("forces_unit") != "eV/angstrom":
            raise ValueError("comparison energy/force units missing or incompatible")
        n = int(record["atom_count"])
        dft_forces = np.asarray(record["forces"], dtype=float)
        forces = np.asarray(predicted["forces"], dtype=float)
        error = float(predicted["energy"]) - float(record["energy"])
        if n <= 0 or forces.shape != (n, 3) or dft_forces.shape != forces.shape or not np.isfinite(forces).all() or not np.isfinite(dft_forces).all() or not np.isfinite(error):
            raise ValueError("invalid paired labels")
        force_error = forces - dft_forces
        comparison.update(status="completed", energy_error=error, energy_error_per_atom=error/n,
                          force_abs_sum=float(np.abs(force_error).sum()),
                          force_square_sum=float((force_error**2).sum()), force_components=n*3)
        record["mlip_prediction"] = {"model_version": predicted["model_version"],
                                     "energy": float(predicted["energy"]), "energy_unit": "eV",
                                     "forces": forces.tolist(), "forces_unit": "eV/angstrom",
                                     "geometry": "DFT_final_frame"}
        state.setdefault("final_frame_dft_errors", []).append({
            "task_id": record["task_id"], "structure_id": record["structure_id"],
            "model_version": record["model_version"], "error_ev_per_atom": abs(error)/n,
            "dft_energy": record["energy"], "mlip_energy": predicted["energy"],
            "comparison": "mlip_vs_dft_on_final_stable_structure"})
    except Exception as error:
        comparison["reason"] = f"{type(error).__name__}: {error}"
    state.setdefault("dft_mlip_comparisons", []).append(comparison)


def export_dft_products(state, directory):
    if directory is None:
        return
    groups = {}
    for record in state.get("dft_dataset_records", []):
        key = json.dumps(record["round_scope"], sort_keys=True)
        groups.setdefault(key, []).append(record)
    exports = {}
    for key, records in groups.items():
        identifier = hashlib.sha256(key.encode()).hexdigest()[:16]
        version = str(records[0].get("model_version") or "unknown-model")
        safe_version = "".join(c if c.isalnum() or c in "-_." else "_" for c in version)
        if safe_version in {".", "..", ""}:
            safe_version = "unknown-model"
        root = Path(directory) / "dft_results" / safe_version / ("DFT-round-" + identifier)
        comparisons = [row for row in state.get("dft_mlip_comparisons", []) if json.dumps(row["round_scope"], sort_keys=True) == key]
        paired = [row for row in comparisons if row["status"] == "completed"]
        metrics = {"round_scope": json.loads(key), "matched_structures": len(paired),
                   "not_evaluated": [row for row in comparisons if row["status"] != "completed"]}
        for field, name, unit in (("energy_error", "energy_total", "eV"),
                                  ("energy_error_per_atom", "energy_per_atom", "eV/atom")):
            values = np.asarray([row[field] for row in paired])
            metrics[name] = {"mae": float(np.abs(values).mean()) if len(values) else None,
                             "rmse": float(np.sqrt((values**2).mean())) if len(values) else None,
                             "mae_unit": unit, "rmse_unit": unit}
        count = sum(row["force_components"] for row in paired)
        metrics["forces"] = {"mae": sum(row["force_abs_sum"] for row in paired)/count if count else None,
                             "rmse": float(np.sqrt(sum(row["force_square_sum"] for row in paired)/count)) if count else None,
                             "mae_unit": "eV/angstrom", "rmse_unit": "eV/angstrom",
                             "components": count, "averaging": "atomic Cartesian components"}
        payloads = {"training.json": [r for r in records if r.get("training_ready") is True],
                    "dft_records.json": records, "mlip_dft_metrics.json": metrics}
        for name, payload in payloads.items():
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            path = root / name
            if path.is_file() and path.read_text(encoding="utf-8") == text:
                continue
            root.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        exports[identifier] = {"directory": str(root), "round_scope": json.loads(key),
                               "records": len(records), "matched_structures": len(paired)}
    state["dft_result_exports"] = exports
