"""Write visible training examples alongside the task result envelope."""

import hashlib
import json
from pathlib import Path

from phase_agent.science.dft.vasp_training_labels import training_records


def write_training_dataset(directory, result):
    outputs = result.get("outputs") or {}
    if "training_frames" not in outputs:
        return
    metadata = {
        key: result[key]
        for key in (
            "task_id",
            "task_key",
            "structure_id",
            "branch_id",
            "model_version",
            "config_version",
            "stage",
            "status",
            "converged",
            "checks_passed",
            "upload_operation_id",
            "search_group_index",
            "parent_relax_round",
        )
        if key in result
    }
    frames = training_records(metadata, outputs)
    for frame in frames:
        # Task-final evidence, not a falsely labelled earlier-frame spin target.
        if outputs.get("magnetic_moments") is not None:
            frame["task_final_magnetic_moments"] = outputs["magnetic_moments"]
    path = Path(directory) / "training.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(frames, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    outputs.update(
        training_file="training.json",
        training_frame_count=len(frames),
        training_checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    result["outputs"] = outputs
