"""Create a small immutable dry-run bundle inside a configured local root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from config_layer.runtime.path_mapping import reject_windows_path_in_remote_text
from execution_layer.step_runner.file_protocol import write_json


def prepare_local_batch_files(*, action, context):
    if (action.get("parameters") or {}).get("mode") == "mc_inputs":
        from execution_layer.local.prepare_mc_upload_batches import prepare_mc_upload_batches
        return prepare_mc_upload_batches(action=action, context=context)
    if _requests_relax_inputs(action, context):
        from execution_layer.local.prepare_relax_upload_batches import prepare_relax_upload_batches
        return prepare_relax_upload_batches(action=action, context=context)
    config = context.get("effective_config") or {}
    root_value = config.get("local_action_directory")
    if not root_value:
        return {"status": "not_configured", "reason": "local_action_directory_missing"}
    parameters = action.get("parameters") or {}
    batch_id = _safe_id(parameters.get("batch_id") or action.get("task_key"))
    task_key = str(action.get("task_key") or "")
    if not task_key:
        raise ValueError("task_key is required")
    root = Path(root_value).resolve()
    target = (root / batch_id).resolve()
    if root not in target.parents:
        raise ValueError("batch directory escaped local_action_directory")
    payload = {
        "schema_version": 1,
        "dry_run": True,
        "submitted": False,
        "batch_id": batch_id,
        "task_id": task_key,
        "config_version": context.get("config_version"),
        "model_version": (context.get("event_state") or {}).get("active_model_version"),
        "target_ids": list(action.get("target_ids") or []),
        "parameters": {key: value for key, value in parameters.items()
                       if key not in {"local_path", "windows_path"}},
    }
    reject_windows_path_in_remote_text(json.dumps(payload, ensure_ascii=False))
    expected = _digest(payload)
    manifest = target / "manifest.json"
    if target.exists():
        if manifest.is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("task_id") == task_key and existing.get("content_checksum") == expected:
                return {"status": "reused", "batch_id": batch_id,
                        "directory": str(target), "manifest_path": str(manifest),
                        "content_checksum": expected}
        raise FileExistsError(f"target already exists and does not match: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload["content_checksum"] = expected
    write_json(manifest, payload)
    notice = target / "DRY_RUN_ONLY.txt"
    notice.write_text("Prepared locally after project approval. Nothing was submitted.\n", encoding="utf-8")
    checksums = target / "SHA256SUMS"
    rows = []
    for path in (manifest, notice):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    checksums.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"status": "prepared", "batch_id": batch_id, "directory": str(target),
            "manifest_path": str(manifest), "checksums_path": str(checksums),
            "content_checksum": expected, "submitted": False}


def _safe_id(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    if not value:
        raise ValueError("batch_id is required")
    return value


def _digest(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _requests_relax_inputs(action, context):
    parameters = action.get("parameters") or {}
    if parameters.get("mode") == "relax_inputs":
        return True
    manager = context.get("manager")
    branches = (getattr(manager, "data", {}) or {}).get("branches", {}) if manager else {}
    targets = action.get("target_ids") or []
    if targets and all(target in branches for target in targets):
        purpose = " ".join(str(action.get(key) or "") for key in ("reason", "expected_purpose"))
        return "relax" in purpose.lower() or "弛豫" in purpose
    return False
