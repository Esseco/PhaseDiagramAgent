"""Validate remote derived phase evidence against the immutable task receipt."""

import json
from pathlib import Path
from phase_agent.tools.remote.integrity import file_checksum


def verified_phase_evidence(path, result_path, task, result):
    try:
        sidecar = json.loads(Path(path).read_text(encoding="utf8"))
        evidence = sidecar["phase_identification"]
        outputs = result.get("outputs") or {}
        structure = Path(outputs.get("structure_path") or "")
        if (
            sidecar.get("schema") != 1
            or sidecar.get("task_id") != task.get("task_id")
            or sidecar.get("task_checksum") != task.get("task_checksum")
            or sidecar.get("result_checksum") != file_checksum(result_path)
            or not isinstance(evidence, dict)
            or evidence.get("origin") != "remote_cpu"
            or not evidence.get("portable_classification_scope")
            or not structure.is_file()
            or evidence.get("structure_sha256") != file_checksum(structure)
        ):
            return None
        return evidence
    except (OSError, ValueError, TypeError, KeyError):
        return None
