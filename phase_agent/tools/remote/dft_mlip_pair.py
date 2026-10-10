"""Validate a transported original-model prediction on the exact DFT final frame."""

from pathlib import Path
import json
import numpy as np
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.science.structures.load_result_structure import load_result_structure


def validate_prediction(result, prediction, expected_digest=None):
    for key in ("task_id", "task_key", "model_version"):
        if not result.get(key) or prediction.get(key) != result[key]:
            raise ValueError(f"MLIP comparison {key} mismatch")
    if prediction.get("status") != "completed":
        raise ValueError(prediction.get("error") or "remote MLIP prediction not completed")
    outputs = result.get("outputs") or {}
    structure, digest = load_result_structure(outputs)
    if prediction.get("structure_sha256") != digest or prediction.get(
        "final_frame_index"
    ) != outputs.get("final_frame_index"):
        raise ValueError("MLIP comparison DFT final-frame mismatch")
    fingerprint = prediction.get("model_sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise ValueError("MLIP comparison model fingerprint unavailable")
    if expected_digest and fingerprint != expected_digest:
        raise ValueError("original-round model fingerprint mismatch")
    forces = np.asarray(prediction.get("forces"), dtype=float)
    if (
        prediction.get("energy_unit") != "eV"
        or prediction.get("forces_unit") != "eV/angstrom"
        or forces.shape != (len(structure), 3)
        or not np.isfinite(forces).all()
        or not np.isfinite(float(prediction.get("energy")))
    ):
        raise ValueError("invalid MLIP comparison energy/forces/units")
    return prediction


def load_pair(result, directory):
    outputs = result.get("outputs") or {}
    if not outputs.get("mlip_result_file"):
        return None  # Legacy DFT remains usable, but does not invent a prediction.
    if outputs["mlip_result_file"] != "mlip_result.json":
        raise ValueError("MLIP result must be mlip_result.json within results directory")
    path = Path(directory) / "mlip_result.json"
    if not path.is_file() or outputs.get("mlip_result_checksum") != file_checksum(path):
        raise ValueError("MLIP result missing or checksum mismatch")
    prediction = json.loads(path.read_text(encoding="utf-8"))
    if prediction.get("status") == "completed":
        validate_prediction(result, prediction)
    else:
        for key in ("task_id", "task_key", "model_version"):
            if prediction.get(key) != result.get(key):
                raise ValueError(f"MLIP failure identity {key} mismatch")
    return prediction
