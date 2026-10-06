"""Training records and round-scoped paired DFT/MLIP error products."""
from copy import deepcopy

import numpy as np

from analysis_layer.feedback.export_dft_products import export_dft_products


def record_dft_products(state, result, evaluator=None, manager=None, *, refresh=False):
    if result.get("stage") not in {"dft_single_point", "dft_relax"} or result.get("status") not in {"completed", "failed", "timeout"}:
        return
    from scientific_layer.dft.spin_acceptance import apply_dft_spin_standard
    result = apply_dft_spin_standard(result, manager)
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
              "status": result.get("status"), "converged": result.get("converged"),
              "checks_passed": result.get("checks_passed", True),
              "energy": outputs.get("training_energy", outputs.get("energy")),
              "energy_unit": outputs.get("energy_unit"), "round_scope": scope,
              **{key: deepcopy(outputs[key]) for key in (
                  "structure", "composition", "atom_count", "forces", "forces_unit",
                  "stress", "stress_unit", "stress_convention", "training_ready",
                  "training_schema", "training_frame_index", "training_energy_kind",
                  "final_frame_index", "final_frame_valid", "magnetic_moments", "magnetic_check", "spin_state_check",
                  "actual_phase", "phase_identification", "structure_path", "remote_mlip_prediction",
                  "mlip_result_file", "mlip_result_checksum") if key in outputs}}
    previous = next((row for row in state.get("dft_dataset_records", []) if row.get("task_id") == record["task_id"]), {})
    previous_comparison = next((row for row in state.get("dft_mlip_comparisons", []) if row.get("task_id") == record["task_id"]), {})
    _upsert_task_record(state, "dft_dataset_records", record)
    from scientific_layer.dft.vasp_training_labels import training_records
    examples = training_records(record, outputs)
    history = state.setdefault("dft_training_records", [])
    known = {r.get("data_id") for r in history}
    if refresh:
        # Retain original frame labels; only refresh task-level diagnostics.
        previously_eligible = {row.get("data_id") for row in history if row.get("task_id") == record["task_id"]
                               and row.get("checks_passed", True) is True}
        pending = state.setdefault("new_dft_records", [])
        previously_pending = {row.get("data_id") for row in pending if row.get("task_id") == record["task_id"]}
        for frame in history:
            if frame.get("task_id") == record["task_id"]:
                for key in ("checks_passed", "actual_phase", "phase_identification", "magnetic_moments", "magnetic_check", "spin_state_check"):
                    if key in record:
                        frame[key] = deepcopy(record[key])
        pending[:] = [row for row in pending if row.get("task_id") != record["task_id"]]
        pending.extend(deepcopy(row) for row in history if row.get("task_id") == record["task_id"]
                       and row.get("checks_passed", True) is True
                       and (row.get("data_id") in previously_pending or row.get("data_id") not in previously_eligible))
    for training_record in examples:
        if training_record["data_id"] not in known:
            history.append(training_record)
            if training_record.get("checks_passed", True) is True:
                state.setdefault("new_dft_records", []).append(deepcopy(training_record))
            known.add(training_record["data_id"])
    comparison = {"task_id": record["task_id"], "round_scope": scope, "status": "not_evaluated"}
    try:
        if record["checks_passed"] is not True:
            raise ValueError("DFT quality/spin standard not passed; final-frame metrics excluded")
        if result.get("status") != "completed" or result.get("converged") is not True:
            raise ValueError("calculation not completed/converged; final-frame metrics excluded")
        if outputs.get("final_frame_valid") is False:
            raise ValueError("actual final frame invalid; final-frame metrics excluded")
        if outputs.get("structure") is not None:
            from scientific_layer.structures.load_result_structure import load_result_structure
            load_result_structure(outputs)
        if outputs.get("training_ready") is not True:
            raise ValueError(outputs.get("training_error") or "portable DFT labels unavailable")
        if refresh and previous_comparison.get("status") == "completed" and previous.get("mlip_prediction"):
            record["mlip_prediction"] = deepcopy(previous["mlip_prediction"])
            _upsert_task_record(state, "dft_mlip_comparisons", deepcopy(previous_comparison))
            return
        if not callable(evaluator):
            raise ValueError("same-frame MLIP evaluator unavailable")
        predicted = evaluator(result=deepcopy(result), structure_id=record["structure_id"], manager=manager)
        if not isinstance(predicted, dict) or predicted.get("model_version") != record["model_version"]:
            raise ValueError("comparison requires matching explicit model_version")
        binding = (state.get("model_registry") or {}).get(record["model_version"]) or {}
        known_digest = binding.get("comparison_model_sha256")
        if known_digest and predicted.get("model_sha256") != known_digest:
            raise ValueError("original-round model fingerprint mismatch")
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
                                     "model_sha256": predicted.get("model_sha256"),
                                     "comparison_model_path": predicted.get("comparison_model_path"),
                                     "energy": float(predicted["energy"]), "energy_unit": "eV",
                                     "forces": forces.tolist(), "forces_unit": "eV/angstrom",
                                     "geometry": "DFT_final_frame"}
        comparison["model_version"] = predicted["model_version"]
        comparison["model_sha256"] = predicted.get("model_sha256")
        if predicted.get("model_sha256"):
            binding = state.setdefault("model_registry", {}).setdefault(record["model_version"], {
                "model": {"version": record["model_version"]}, "status": "comparison_reference"})
            binding.setdefault("comparison_model_sha256", predicted["model_sha256"])
        state.setdefault("final_frame_dft_errors", []).append({
            "task_id": record["task_id"], "structure_id": record["structure_id"],
            "model_version": record["model_version"], "error_ev_per_atom": abs(error)/n,
            "dft_energy": record["energy"], "mlip_energy": predicted["energy"],
            "comparison": "mlip_vs_dft_on_final_stable_structure"})
    except Exception as error:
        comparison["reason"] = f"{type(error).__name__}: {error}"
    _upsert_task_record(state, "dft_mlip_comparisons", comparison)
    if refresh and comparison["status"] != "completed":
        state["final_frame_dft_errors"] = [row for row in state.get("final_frame_dft_errors", []) if row.get("task_id") != record["task_id"]]


def _upsert_task_record(state, key, record):
    rows = state.setdefault(key, [])
    index = next((i for i, row in enumerate(rows) if row.get("task_id") == record.get("task_id")), None)
    if index is None:
        rows.append(record)
    else:
        rows[index] = record
