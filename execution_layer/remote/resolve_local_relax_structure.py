"""Require a downloaded final MLIP structure before accepting its result."""
from copy import deepcopy
from pathlib import Path


def resolve_local_relax_structure(result, task_directory):
    payload = deepcopy(result)
    outputs = payload.get("outputs") or {}
    raw = outputs.get("structure_path") or outputs.get("final_structure_path")
    if not raw:
        return None
    path = Path(str(raw).replace("\\", "/"))
    root = Path(task_directory).resolve()
    candidates = [root / path.name, root / "pool" / path.name]
    for candidate in candidates:
        if candidate.is_file() and root in candidate.resolve().parents:
            expected = outputs.get("structure_checksum")
            if expected:
                from execution_layer.remote.integrity import file_checksum
                if file_checksum(candidate) != expected:
                    return None
            payload["outputs"] = {**outputs, "structure_path": str(candidate.resolve())}
            payload["result_path"] = str(candidate.resolve())
            return payload
    return None
