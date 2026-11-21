"""Allow only verified, disjoint frozen DFT planning beside unsubmitted inputs."""

import hashlib
import json
from pathlib import Path


def _file_geometry(path, digest):
    from phase_agent.analysis.convergence.independent_evidence import geometry_key

    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return None
    return geometry_key({"structure": str(path)})


def allowed(frame):
    """Permit existing approval flow, never modify or waive the waiting tasks."""
    state = (frame.get("feedback") or {}).get("state") or {}
    wait = frame.get("manual_wait") or {}
    action = getattr(frame.get("agent_client"), "_frozen_dft_action", None)
    feedback = frame.get("human_feedback")
    if action is None and isinstance(feedback, dict) and feedback.get("decision") == "approve":
        record = (state.get("pending_execution_policies") or {}).get(frame.get("invocation_id")) or {}
        proposal = record.get("agent_proposal") or {}
        action = proposal.get("raw_action") or proposal
    if (
        not isinstance(action, dict)
        or action.get("tool") != "select_dft_candidates"
        or (action.get("_decision_team") or {}).get("backend") != "codex_native"
        or not action.get("_simulation_plan")
        or not (action.get("parameters") or {}).get("prepare_inputs_after_approval")
        or set(wait.get("waiting_by_stage") or {}) != {"dft_single_point"}
    ):
        return False
    decisions = (action.get("parameters") or {}).get("decisions") or []
    if not isinstance(decisions, list) or any(
        not isinstance(d, dict) or d.get("action") == "DFT_RELAX" for d in decisions
    ):
        return False
    selected = [d.get("candidate_id") for d in decisions if d.get("action") == "DFT_SINGLE_POINT"]
    targets = action.get("target_ids") or []
    waiting_ids = wait.get("task_ids") or []
    if (
        not selected or len(selected) != len(set(selected)) or set(selected) != set(targets)
        or len(targets) != len(set(targets)) or not waiting_ids
        or len(waiting_ids) != len(set(waiting_ids))
        or wait.get("waiting_task_count") != len(waiting_ids)
        or wait.get("waiting_by_stage") != {"dft_single_point": len(waiting_ids)}
    ):
        return False
    tasks = state.get("tasks") or []
    by_task = {t.get("task_id"): t for t in tasks}
    candidates = state.get("qbc_candidates") or []
    by_candidate = {c.get("candidate_id"): c for c in candidates}
    if len(by_task) != len(tasks) or len(by_candidate) != len(candidates):
        return False
    model = state.get("active_model_version")
    if not model:
        return False
    try:
        old_geometries = set()
        for tid in waiting_ids:
            task = by_task.get(tid) or {}
            if task.get("stage") != "dft_single_point" or task.get("status") != "pending":
                return False
            spec = json.loads(Path(task["input_path"]).read_text(encoding="utf8"))
            digest = task.get("input_structure_sha256")
            if (not digest or not task.get("task_checksum") or spec.get("task_id") != tid or spec.get("task_checksum") != task.get("task_checksum")
                or spec.get("input_structure_sha256") != digest):
                return False
            key = _file_geometry(spec["structure_path"], digest)
            if not key:
                return False
            old_geometries.add(key)
        new_geometries = set()
        for cid in selected:
            candidate = by_candidate.get(cid) or {}
            digest = candidate.get("structure_sha256")
            if (not digest or candidate.get("model_version") != model
                or candidate.get("qbc_model_version") != model
                or candidate.get("qbc_frame_sha256") != digest
                or (candidate.get("qbc") or {}).get("status") != "completed"):
                return False
            key = _file_geometry(candidate["structure_path"], digest)
            if not key or key in old_geometries or key in new_geometries:
                return False
            new_geometries.add(key)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    from phase_agent.agents.reliability.selection.dft_round_policy import dft_task_limit

    try:
        return len(selected) <= dft_task_limit(frame.get("effective_config") or {})
    except ValueError:
        return False
