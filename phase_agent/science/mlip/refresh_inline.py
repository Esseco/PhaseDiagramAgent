"""Single-task prediction and force-triggered relaxation with restartable evidence."""

import hashlib
import json
import math
from pathlib import Path


def run_inline_refresh(job, predictor, relaxer):
    source = Path(job["structure_path"])
    params = job.get("parameters") or {}
    threshold = params.get("refresh_force_threshold")
    if isinstance(threshold, bool) or threshold not in (0.1, 0.2):
        raise ValueError("invalid_refresh_force_threshold")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = {
        "structure_sha256": digest,
        "model_version": job.get("model_version"),
        "model_path": job.get("model_path"),
        "model_paths": job.get("model_paths"),
        "parameters": params,
    }
    if not identity["model_version"]:
        raise ValueError("refresh_model_identity_missing")
    output = Path(job["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "refresh_prediction.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf8"))
        if saved.get("identity") != identity:
            raise ValueError("refresh_prediction_identity_mismatch")
        prediction = saved["prediction"]
    else:
        prediction = predictor(job)
        if (
            prediction.get("status") != "completed"
            or prediction.get("single_point_completed") is not True
        ):
            raise ValueError("refresh_prediction_incomplete")
    if (
        prediction.get("structure_checksum") != digest
        or prediction.get("mlip_version") != identity["model_version"]
    ):
        raise ValueError("refresh_prediction_frame_or_model_mismatch")
    # The final relaxed prediction may overwrite evaluated.vasp; retain the original
    # task input path as the provenance of these initial same-frame forces.
    prediction = {**prediction, "structure_path": str(source.resolve())}
    forces = prediction.get("forces")
    if prediction.get("forces_unit") != "eV/angstrom" or not isinstance(forces, list) or not forces:
        raise ValueError("refresh_prediction_forces_missing")
    if any(
        not isinstance(row, (list, tuple))
        or len(row) != 3
        or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
            for x in row
        )
        for row in forces
    ):
        raise ValueError("refresh_prediction_forces_invalid")
    if prediction.get("atom_count") != len(forces):
        raise ValueError("refresh_prediction_force_count_mismatch")
    maximum = max(math.sqrt(sum(x * x for x in row)) for row in forces)
    if not cache.exists():
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"identity": identity, "prediction": prediction}, ensure_ascii=False),
            encoding="utf8",
        )
        temporary.replace(cache)
    evidence = {
        "initial_prediction": prediction,
        "force_max_ev_per_angstrom": maximum,
        "threshold_ev_per_angstrom": threshold,
        "input_structure_sha256": digest,
    }
    if maximum <= threshold:
        return {**prediction, "refresh_force_followup": {**evidence, "relaxed": False}}
    relaxed = relaxer({**job, "operation": "relax"})
    if relaxed.get("status") != "completed" or not isinstance(relaxed.get("optimizer_converged"), bool):
        raise ValueError("refresh_relaxation_result_incomplete")
    return {**relaxed, "converged": relaxed['optimizer_converged'],
            "refresh_force_followup": {**evidence, "relaxed": True}}
