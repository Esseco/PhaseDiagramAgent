"""Revalidate a known static acceptance failure in a new recovery workspace only."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.legacy_recovery_runner import safe, terminal
from phase_agent.tools.remote.predict_dft_final_frame import supplement_results


def reassessment_transform(work, options, *, validator=None, predictor=None):
    """Keep original task outputs immutable; reuse the existing prediction adapter."""
    from Process_Vasp.workflows.atomate_runner import _same_structure, _validate
    from pymatgen.core import Structure

    validate = validator or _validate
    bindings = options["eligible_tasks"]

    def transform(destination, source, task, original):
        tid = task["expected"]["task_id"]
        print(f"[{tid}] checking existing DFT result", flush=True)
        if tid not in bindings:
            print(f"[{tid}] skipped: no verified converged final frame", flush=True)
            return {"status": "skipped", "reason": "not_selected_for_reassessment"}
        bound = bindings[tid]
        original_bytes = (destination / "result.json").read_bytes()
        original_sha = hashlib.sha256(original_bytes).hexdigest()
        if original_sha != bound["original_result_sha256"]:
            raise ValueError("reassessment_original_result_changed")
        if (
            original.get("stage") != "dft_single_point"
            or original.get("status") != "failed"
        ):
            raise ValueError("reassessment_requires_known_failed_static")
        outputs = original.get("outputs") or {}
        if not all(
            outputs.get(key) is True
            for key in (
                "xml_complete",
                "electronic_converged",
                "final_frame_valid",
                "training_ready",
            )
        ):
            raise ValueError("reassessment_final_frame_not_verified")
        worker = safe(source, task["input_directory"])
        config = json.loads((worker / "workflow.json").read_text(encoding="utf-8"))
        checkpoint = json.loads(
            (worker / "workflow_state.json").read_text(encoding="utf-8")
        )
        entry = (checkpoint.get("stages") or {}).get("static") or {}
        if (
            config.get("calculation") != "static"
            or entry.get("error") != "static unexpectedly changed the structure"
        ):
            raise ValueError("reassessment_not_the_known_geometry_failure")
        initial_path = safe(worker, config["structure_file"])
        comparison_path = worker / "comparison_model.json"
        if (
            file_checksum(initial_path) != bound["initial_structure_sha256"]
            or file_checksum(comparison_path) != bound["comparison_model_sha256"]
        ):
            raise ValueError("reassessment_original_input_changed")
        model = json.loads(comparison_path.read_text(encoding="utf-8"))
        if model != options["model"] or model.get(
            "version", model.get("name")
        ) != original.get("model_version"):
            raise ValueError("reassessment_original_model_changed")
        model_sha = file_checksum(Path(model["model_path"]))
        if model.get("model_sha256") and model_sha != model["model_sha256"]:
            raise ValueError("reassessment_original_model_fingerprint_mismatch")
        calculation = safe(worker, entry["directory"])
        # Read and validate complete existing XML/OUTCAR/CONTCAR. Never run_workflow.
        final = validate(calculation, "static")
        initial = Structure.from_file(initial_path)
        saved = Structure.from_dict(outputs["structure"])
        if not _same_structure(initial, final) or not _same_structure(saved, final):
            raise ValueError("reassessment_static_geometry_really_changed")
        cache_key = hashlib.sha256((original_sha + model_sha).encode()).hexdigest()
        cache = Path(work) / "reassessment_cache" / tid
        cache.mkdir(parents=True, exist_ok=True)
        cached_path = cache / "result.json"
        cached = (
            json.loads(cached_path.read_text(encoding="utf-8"))
            if cached_path.is_file()
            else {}
        )
        if (cached.get("recovery_provenance") or {}).get("cache_key") != cache_key:
            derived = deepcopy(original)
            derived.update(status="completed", converged=True)
            derived.pop("error", None)
            derived["outputs"]["converged"] = True
            derived["recovery_provenance"] = {
                "mode": "static_geometry_reassessment_v1",
                "original_status": original["status"],
                "original_error": original.get("error"),
                "original_result_sha256": original_sha,
                "original_workflow_error": entry["error"],
                "geometry_check": "ordered_species_lattice_and_periodic_coordinates",
                "cache_key": cache_key,
            }
            _write(cached_path, derived)
            receipt = json.loads(
                (destination / "task.finished.json").read_text(encoding="utf-8")
            )
            receipt.update(
                status="completed", result_checksum=file_checksum(cached_path)
            )
            _write(cache / "task.finished.json", receipt)
        print(
            f"[{tid}] geometry verified; supplementing original-model same-frame prediction",
            flush=True,
        )
        reports = supplement_results(cache, model, predictor=predictor)
        if len(reports) != 1:
            raise ValueError("reassessment_prediction_scope_invalid")
        # Original evidence stays alongside the new derived result, without duplication
        # of large DFT files or rewriting the original remote receipt.
        (destination / "original_result.json").write_bytes(original_bytes)
        shutil.copyfile(
            destination / "task.finished.json",
            destination / "original_task.finished.json",
        )
        for filename in ("result.json", "task.finished.json", "mlip_result.json"):
            shutil.copyfile(cache / filename, destination / filename)
        revised = terminal(destination, task["expected"])
        if revised is None:
            raise ValueError("reassessment_receipt_invalid")
        report = reports[0]
        status = "prediction_failed" if report["status"] == "failed" else "completed"
        print(f"[{tid}] {status} ({report['status']})", flush=True)
        return {
            "status": status,
            "prediction": report["status"],
            "error": report.get("error"),
            "original_result_sha256": original_sha,
        }

    return transform


def _write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
