"""Verify an explicitly authorized independent DFT scope for unknown GS frames."""

from hashlib import sha256
from pathlib import Path


def verify_unknown_gs_frames(state, frame_hashes, *, allowed_pending_task_ids=frozenset()):
    """Verify current facts; never change unknown into a normal monitor result."""
    from phase_agent.evidence.dft_extrapolation import candidate_extrapolation
    from phase_agent.evidence.model_diagram_scope import selection_diagram
    from phase_agent.analysis.convergence.independent_evidence import geometry_key

    if not isinstance(frame_hashes, dict) or not 1 <= len(frame_hashes) <= 100:
        raise ValueError("unknown_gs_scope_invalid")
    if len(set(frame_hashes.values())) != len(frame_hashes):
        raise ValueError("unknown_gs_duplicate_frame")
    model = state.get("active_model_version")
    diagram = selection_diagram(state)
    if not model or diagram.get("status") != "completed" or diagram.get("model_version") != model:
        raise ValueError("unknown_gs_hull_unavailable")
    prior = (state.get("exact_candidate_relax_authorization") or {}).get(
        "candidate_frame_sha256"
    ) or {}
    candidates = {}
    for row in state.get("qbc_candidates") or []:
        cid = row.get("candidate_id")
        if cid in frame_hashes:
            if cid in candidates:
                raise ValueError("unknown_gs_duplicate_candidate")
            candidates[cid] = row
    eligible = {}
    for entry in diagram.get("entries") or []:
        if entry.get("is_stable"):
            key = (entry.get("structure_id"), entry.get("structure_sha256"))
            eligible.setdefault(key, set()).add(entry.get("source_version") or model)

    completed = set()
    for row in state.get("dft_dataset_records") or []:
        if row.get("status") == "completed" and row.get("checks_passed") is True:
            key = geometry_key(row)
            if key:
                completed.add(key)
    for cid, digest in frame_hashes.items():
        row = candidates.get(cid) or {}
        if not isinstance(cid, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("unknown_gs_invalid_identity")
        if prior.get(cid) != digest or row.get("structure_sha256") != digest:
            raise ValueError("unknown_gs_candidate_binding:" + cid)
        path = Path(row.get("structure_path") or "")
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("unknown_gs_checksum_mismatch:" + cid)
        sources = eligible.get((row.get("structure_id"), digest)) or set()
        if not sources:
            raise ValueError("unknown_gs_not_stable:" + cid)
        source_model = row.get("energy_source_version") or row.get("model_version")
        if source_model not in sources or (
            source_model != model
            and not (
                diagram.get("mixed_model_versions") is True
                and diagram.get("energy_policy") == "latest_valid_per_structure_v1"
            )
        ):
            raise ValueError("unknown_gs_energy_source_mismatch:" + cid)
        if (
            candidate_extrapolation({**row, "model_version": model}, state).get("status")
            != "unknown"
        ):
            raise ValueError("unknown_gs_monitor_not_unknown:" + cid)
        key = geometry_key({"structure": str(path)})
        if not key or key in completed:
            raise ValueError("unknown_gs_existing_dft_or_geometry_invalid:" + cid)
        if any(
            t.get("stage") in {"dft_single_point", "dft_relax"}
            and t.get("status") in {"pending", "running", "completed"}
            and t.get("input_structure_sha256") == digest
            and t.get("task_id") not in allowed_pending_task_ids
            for t in state.get("tasks") or []
        ):
            raise ValueError("unknown_gs_existing_dft_task:" + cid)
    return diagram


def is_authorized_unknown_gs_scope(state, decisions):
    """Return true only for the entire exact authorized single-point scope."""
    auth = state.get("unknown_gs_dft_authorization") or {}
    frames = auth.get("candidate_frame_sha256") or {}
    if (
        auth.get("status") != "approved_for_preparation"
        or not frames
        or auth.get("model_version") != state.get("active_model_version")
        or auth.get("config_version") != state.get("confirmed_config_version")
    ):
        return False
    if not isinstance(decisions, list) or len(decisions) != len(frames):
        return False
    if any(not isinstance(d, dict) or d.get("action") != "DFT_SINGLE_POINT" for d in decisions):
        return False
    ids = [d.get("candidate_id") for d in decisions]
    if len(set(ids)) != len(ids) or set(ids) != set(frames):
        return False
    try:
        diagram = verify_unknown_gs_frames(state, frames)
        return diagram.get("version") == auth.get("hull_version")
    except (ValueError, OSError, TypeError, KeyError):
        return False


def is_authorized_unknown_gs_tasks(state, tasks, *, approval_record_id=None, approved_action=None):
    """Recheck an approved exact scope for input creation, including partial recovery."""
    auth = state.get("unknown_gs_dft_authorization") or {}
    frames = auth.get("candidate_frame_sha256") or {}
    if (
        auth.get("status") != "approved_for_preparation"
        or not frames
        or auth.get("model_version") != state.get("active_model_version")
        or auth.get("config_version") != state.get("confirmed_config_version")
        or not isinstance(tasks, list)
        or len(tasks) != len(frames)
    ):
        return False
    ids = [t.get("task_id") for t in tasks if isinstance(t, dict)]
    cids = [t.get("object_id") or t.get("structure_id") for t in tasks if isinstance(t, dict)]
    if (
        len(ids) != len(tasks)
        or None in ids
        or len(set(ids)) != len(ids)
        or set(cids) != set(frames)
        or len(set(cids)) != len(cids)
    ):
        return False
    parents = {t.get("parent_decision_id") for t in tasks}
    if len(parents) != 1 or not next(iter(parents), None):
        return False
    parent = next(iter(parents))
    if approval_record_id == parent and isinstance(approved_action, dict):
        action = approved_action
    else:
        record = next(
            (r for r in state.get("action_records") or [] if r.get("record_id") == parent), {}
        )
        if (record.get("human_feedback") or {}).get("decision") != "approve":
            return False
        action = (record.get("agent_proposal") or {}).get("raw_action") or {}
    decisions = (action.get("parameters") or {}).get("decisions") or []
    if (
        action.get("tool") != "select_dft_candidates"
        or len(decisions) != len(frames)
        or {d.get("candidate_id") for d in decisions if isinstance(d, dict)} != set(frames)
        or any(not isinstance(d, dict) or d.get("action") != "DFT_SINGLE_POINT" for d in decisions)
    ):
        return False
    for task, cid in zip(tasks, cids):
        if (
            task.get("stage") != "dft_single_point"
            or task.get("status") != "pending"
            or task.get("slurm_batch_id")
            or task.get("input_structure_sha256") != frames[cid]
            or task.get("model_version") != auth.get("model_version")
            or task.get("config_version") != auth.get("config_version")
            or task.get("scientific_parent_task_key") != action.get("task_key")
        ):
            return False
        path = Path(task.get("structure_path") or "")
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != frames[cid]:
            return False
    try:
        diagram = verify_unknown_gs_frames(state, frames, allowed_pending_task_ids=frozenset(ids))
        return diagram.get("version") == auth.get("hull_version")
    except (ValueError, OSError, TypeError, KeyError):
        return False
