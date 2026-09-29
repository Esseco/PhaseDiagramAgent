"""Content checks used at the local/remote trust boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def payload_checksum(payload) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_checksum(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_result(result_path, marker_path, expected: dict) -> dict:
    result_path, marker_path = Path(result_path), Path(marker_path)
    if not result_path.is_file() or not marker_path.is_file():
        return {"valid": False, "reason": "result_or_completion_marker_missing"}
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"valid": False, "reason": f"invalid_json:{error}"}
    for key in ("task_id", "task_key", "batch_id", "config_version", "model_version", "protocol_version",
                "input_file_version"):
        wanted = expected.get(key)
        if wanted is not None and marker.get(key) != wanted:
            return {"valid": False, "reason": f"marker_{key}_mismatch"}
        if wanted is not None and result.get(key) != wanted:
            return {"valid": False, "reason": f"result_{key}_mismatch"}
    if marker.get("task_checksum") != expected.get("task_checksum"):
        return {"valid": False, "reason": "task_checksum_mismatch"}
    if marker.get("result_checksum") != file_checksum(result_path):
        return {"valid": False, "reason": "result_checksum_mismatch"}
    return {"valid": True, "result": result, "marker": marker}
