"""Verify unsubmitted refresh inputs before preserving an abandoned partial plan."""

import hashlib
import json
from pathlib import Path

from phase_agent.tools.remote.integrity import payload_checksum, file_checksum
from phase_agent.tools.state.execution_receipts import execution_events
from phase_agent.tools.workflows.model_refresh_state import fingerprint


def inspect_abandoned_refresh(state_path, state, issue):
    identity = issue.get("identity") or {}
    invocation = identity.get("invocation_id")
    if (issue.get("phase") != "returned" or issue.get("tool_outcome") != "completed"
            or identity.get("tool") != "prepare_local_batch_files"):
        raise ValueError("refresh_staging_receipt_invalid")
    events = execution_events(state_path, invocation)
    claimed = [e["payload"] for e in events if e["event"] == "claimed"]
    returned = [e["payload"] for e in events if e["event"] == "tool_returned"]
    if len(claimed) != 1 or len(returned) != 1:
        raise ValueError("refresh_staging_events_invalid")
    if claimed[0].get("identity") != identity or returned[0].get("identity") != identity or returned[0].get("outcome") != "completed":
        raise ValueError("refresh_staging_event_identity_invalid")
    action = claimed[0].get("action") or {}
    evidence = returned[0].get("evidence") or {}
    params = action.get("parameters") or {}
    plan = params.get("refresh_preview") or {}
    checksum = plan.get("checksum")
    if (params.get("mode") != "model_refresh_inputs" or action.get("task_key") != invocation
            or params.get("preview_checksum") != checksum
            or not checksum or fingerprint({k: v for k, v in plan.items() if k != "checksum"}) != checksum
            or plan.get("refresh_scope") != "frozen_baseline_hull_vertices"
            or action.get("config_version") != state.get("confirmed_config_version")
            or plan.get("new_model_version") != state.get("active_model_version")
            or evidence.get("result_status") != "prepared"):
        raise ValueError("refresh_staging_approval_binding_invalid")
    pending = [row for row in (state.get("pending_execution_policies") or {}).values()
               if ((row.get("agent_proposal") or {}).get("raw_action") or {}).get("task_key") == invocation]
    if len(pending) != 1:
        raise ValueError("refresh_staging_pending_approval_invalid")
    approved = pending[0]["agent_proposal"]["raw_action"]
    if any(approved.get(k) != action.get(k) for k in ("tool", "task_key", "config_version", "budget", "target_ids", "parameters")):
        raise ValueError("refresh_staging_pending_approval_invalid")
    rows = plan.get("candidates") or []
    refs = evidence.get("returned_task_refs") or []
    by_hash = {row.get("structure_sha256"): row for row in rows}
    if not rows or len(by_hash) != len(rows) or len(refs) != len(rows):
        raise ValueError("refresh_staging_frame_count_invalid")
    task_ids = {row.get("task_id") for row in refs}
    if None in task_ids or len(task_ids) != len(refs):
        raise ValueError("refresh_staging_task_identity_invalid")
    expected_artifacts = {str(ref.get(k)) for ref in refs for k in ("input_path", "result_path")}
    if {item.get("artifact_path") for item in (evidence.get("artifacts") or [])} != expected_artifacts:
        raise ValueError("refresh_staging_artifact_scope_invalid")
    if any(t.get("task_id") in task_ids or t.get("parent_decision_id") == invocation
           for t in state.get("tasks") or []):
        raise ValueError("refresh_staging_has_registered_tasks")
    if any((v or {}).get("status") in {"reserved", "submitted"}
           for k, v in (state.get("budget_reservations") or {}).items()
           if k.startswith("model-refresh:" + checksum + ":")):
        raise ValueError("refresh_staging_has_budget_reservations")
    root = Path(state_path).resolve().parent.parent / "submissions"
    batch_dirs = set()
    file_hashes = []
    seen = set()
    for ref in refs:
        path = Path(ref.get("input_path") or "").resolve()
        if not path.is_relative_to(root.resolve()) or path.name != "task.json" or not path.is_file():
            raise ValueError("refresh_staging_task_path_invalid")
        task = json.loads(path.read_text(encoding="utf-8"))
        content_hash = task.get("input_structure_sha256")
        row = by_hash.get(content_hash)
        if row is None or content_hash in seen:
            raise ValueError("refresh_staging_geometry_scope_invalid")
        seen.add(content_hash)
        if (any(task.get(k) != ref.get(k) for k in ("task_id", "task_key", "model_version", "config_version", "parent_decision_id"))
                or task.get("structure_id") != row.get("structure_id")
                or task.get("model_refresh_id") != checksum or task.get("status") != "pending"
                or task.get("task_key") != f"model-refresh:{checksum}:0:{content_hash}:{row.get('operation')}"
                or task.get("task_checksum") != payload_checksum({k:v for k,v in task.items() if k != "task_checksum"})
                or not (path.parent / "initial.vasp").is_file()
                or file_checksum(path.parent / "initial.vasp") != content_hash
                or Path(ref.get("result_path") or "").exists()):
            raise ValueError("refresh_staging_task_binding_invalid")
        batch_dirs.add(path.parent.parent)
        file_hashes.append((str(path), file_checksum(path), content_hash))
    if seen != set(by_hash):
        raise ValueError("refresh_staging_frames_missing")
    manifests = []
    manifest_ids = set()
    for directory in sorted(batch_dirs):
        manifest_path, snapshot_path = directory / "manifest.json", directory / "batch.snapshot.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if (snapshot.get("manifest_checksum") != payload_checksum(manifest)
                or set(snapshot.get("task_ids") or []) != {r.get("task_id") for r in manifest}
                or not set(snapshot.get("task_ids") or []).issubset(task_ids)
                or snapshot.get("config_version") != state.get("confirmed_config_version")
                or snapshot.get("model_version") != state.get("active_model_version")
                or not (directory / "GPU.sh").is_file()):
            raise ValueError("refresh_staging_manifest_invalid")
        for entry in manifest:
            tid = entry.get("task_id")
            ref = next((item for item in refs if item.get("task_id") == tid), None)
            task_path = (directory / (entry.get("input_path") or "")).resolve()
            if (tid in manifest_ids or ref is None or task_path != Path(ref["input_path"]).resolve()
                    or not task_path.is_relative_to(directory)
                    or entry.get("task_key") != ref.get("task_key")
                    or entry.get("config_version") != ref.get("config_version")
                    or entry.get("model_version") != ref.get("model_version")
                    or entry.get("task_checksum") != json.loads(task_path.read_text(encoding="utf-8")).get("task_checksum")):
                raise ValueError("refresh_staging_manifest_task_invalid")
            manifest_ids.add(tid)
        manifests.append((str(manifest_path), file_checksum(manifest_path), file_checksum(snapshot_path),
                          file_checksum(directory / "GPU.sh")))
    if manifest_ids != task_ids:
        raise ValueError("refresh_staging_manifest_tasks_missing")
    if any(set(batch.get("task_ids") or []) & task_ids for batch in state.get("slurm_batches") or []):
        raise ValueError("refresh_staging_has_registered_batch")
    digest = hashlib.sha256(json.dumps([sorted(file_hashes), manifests], sort_keys=True).encode()).hexdigest()
    return {"kind": "abandoned_unsubmitted_refresh_staging", "submitted": False,
            "task_count": len(refs), "task_ids": sorted(task_ids), "plan_checksum": checksum,
            "batch_directories": sorted(str(p) for p in batch_dirs), "inventory_sha256": digest}
