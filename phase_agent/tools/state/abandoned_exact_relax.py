"""Inspect an unregistered, incomplete exact-frame Relax input staging batch."""

import hashlib
import json
from pathlib import Path

from phase_agent.tools.remote.integrity import payload_checksum
from phase_agent.tools.state.execution_receipts import execution_events


def inspect_abandoned_exact_relax(state_path, state, issue, directory):
    identity = issue.get("identity") or {}
    invocation = identity.get("invocation_id")
    events = execution_events(state_path, invocation)
    claimed = [e["payload"] for e in events if e["event"] == "claimed"]
    returned = [e["payload"] for e in events if e["event"] == "tool_returned"]
    if (issue.get("phase") != "returned" or issue.get("tool_outcome") != "failed"
            or len(claimed) != 1 or len(returned) != 1
            or claimed[0].get("identity") != identity or returned[0].get("identity") != identity
            or returned[0].get("outcome") != "failed"
            or (returned[0].get("evidence") or {}).get("error") != "MemoryError: "):
        raise ValueError("exact_staging_receipt_invalid")
    action = claimed[0].get("action") or {}
    params = action.get("parameters") or {}
    hashes = params.get("candidate_frame_sha256") or {}
    auth = state.get("exact_candidate_relax_authorization") or {}
    if (action.get("operation") != "prepare_mlip_relax" or params.get("mode") != "relax_inputs"
            or params.get("selection_scope") != "candidate_ids" or not hashes
            or set(action.get("target_ids") or []) != set(hashes)
            or hashes != auth.get("candidate_frame_sha256")
            or params.get("model_version") != state.get("active_model_version")
            or identity.get("config_version") != state.get("confirmed_config_version")):
        raise ValueError("exact_staging_action_invalid")
    root = (Path(state_path).resolve().parent.parent / "submissions").resolve()
    path = Path(directory).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError("exact_staging_directory_invalid")
    manifest_path, snapshot_path = path / "manifest.json", path / "batch.snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf8"))
    if not manifest or len(manifest) >= len(hashes):
        raise ValueError("exact_staging_not_incomplete")
    checksum = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if snapshot.get("manifest_checksum") != payload_checksum(manifest):
        raise ValueError("exact_staging_manifest_checksum_mismatch")
    if (snapshot.get("model_version") != params.get("model_version")
            or snapshot.get("config_version") != identity.get("config_version")):
        raise ValueError("exact_staging_snapshot_binding_mismatch")
    if any(b.get("batch_id") == snapshot.get("batch_id") for b in state.get("slurm_batches") or []):
        raise ValueError("exact_staging_batch_already_registered")
    ids, cids, files = set(), set(), []
    for entry in manifest:
        task_path = (path / entry.get("input_path", "")).resolve()
        if not task_path.is_relative_to(path) or task_path.name != "task.json":
            raise ValueError("exact_staging_task_path_invalid")
        task = json.loads(task_path.read_text(encoding="utf8"))
        tid, cid = task.get("task_id"), task.get("candidate_id")
        frame = task_path.with_name("initial.vasp")
        result = (path / task.get("result_path", "")).resolve()
        if (tid in ids or cid in cids or cid not in hashes or tid != entry.get("task_id")
                or task.get("parent_decision_id") != invocation or task.get("status") != "pending"
                or task.get("input_structure_sha256") != hashes[cid]
                or not frame.is_file() or checksum(frame) != hashes[cid]
                or task.get("task_checksum") != payload_checksum({k:v for k,v in task.items() if k != "task_checksum"})
                or task.get("model_version") != params.get("model_version")
                or task.get("config_version") != identity.get("config_version")
                or result.is_file() or task.get("job_id") or task.get("slurm_job_id")):
            raise ValueError("exact_staging_task_binding_invalid")
        ids.add(tid); cids.add(cid)
        files.append({"task_id":tid, "candidate_id":cid, "task_sha256":checksum(task_path), "frame_sha256":checksum(frame)})
    if set(snapshot.get("task_ids") or []) != ids:
        raise ValueError("exact_staging_snapshot_task_mismatch")
    if any(t.get("task_id") in ids or t.get("parent_decision_id") == invocation for t in state.get("tasks") or []):
        raise ValueError("exact_staging_has_registered_tasks")
    return {"kind":"abandoned_exact_candidate_relax_staging", "directory":str(path),
            "batch_id":snapshot.get("batch_id"), "submitted":False,
            "manifest_sha256":checksum(manifest_path), "snapshot_sha256":checksum(snapshot_path),
            "frames":sorted(files,key=lambda x:x["candidate_id"]),
            "missing_candidate_ids":sorted(set(hashes)-cids)}
