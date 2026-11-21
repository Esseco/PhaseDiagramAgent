"""Restore an independent DFT draw replaced by a verified rejected proposal."""

from contextlib import closing
from hashlib import sha256
from pathlib import Path
import sqlite3
from threading import Lock

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.agents.strategic.codex_submission_rejection import rejection_checkpoint
from phase_agent.evidence.model_diagram_scope import selection_diagram
from phase_agent.persistence.checkpoint_blobs import (
    CheckpointBlobSerde,
    STATE_FIELDS,
    STATE_FIELDS_V2,
)
from phase_agent.persistence.compact_checkpoints import CompressedSerde
from phase_agent.persistence.sqlite_paths import readonly_uri
from phase_agent.persistence.state.field_index import open_indexed_state
from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.tools.workflows.lifecycle_support import _save_runner_state


def _fields(serde, state, names):
    for marker in (STATE_FIELDS, STATE_FIELDS_V2):
        if set(state) == {marker}:
            return serde._restore_state(
                {marker: {key: ref for key, ref in state[marker].items() if key in names}}
            )
    return {key: state[key] for key in names if key in state}


def _matching_rows(rows):
    by_id = {row.get("candidate_id"): row for row in rows}
    if not all(by_id) or len(by_id) != len(rows):
        raise ValueError("dft_audit_candidate_identity_ambiguous")
    return by_id


def _restore_ids(before, after, current, action):
    old_ids = before.get("dft_audit_candidate_ids") or []
    replaced_ids = after.get("dft_audit_candidate_ids") or []
    targets = action.get("target_ids") or []
    decisions = (action.get("parameters") or {}).get("decisions") or []
    selected = [
        row.get("candidate_id")
        for row in decisions
        if row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    ]
    if (
        action.get("tool") != "select_dft_candidates"
        or not targets
        or len(set(targets)) != len(targets)
        or set(selected) != set(targets)
        or len(selected) != len(targets)
        or not old_ids
        or len(set(old_ids)) != len(old_ids)
        or not replaced_ids
        or len(set(replaced_ids)) != len(replaced_ids)
        or set(targets) & set(replaced_ids)
    ):
        raise ValueError("dft_audit_replacement_not_proven")
    for key in ("active_model_version", "confirmed_config_version"):
        if not before.get(key) or before[key] != current.get(key):
            raise ValueError("dft_audit_model_or_config_changed")
    old_hull = selection_diagram(before)
    current_hull = selection_diagram(current)
    if (
        old_hull.get("status") != "completed"
        or not old_hull.get("version")
        or any(
            current_hull.get(key) != old_hull.get(key)
            for key in ("status", "version", "model_version", "energy_basis_id")
        )
    ):
        raise ValueError("dft_audit_hull_changed")
    original = _matching_rows(before.get("qbc_candidates") or [])
    now = _matching_rows(current.get("qbc_candidates") or [])
    restored = [cid for cid in old_ids if cid in original and cid in now]
    if not set(targets) & set(restored):
        raise ValueError("dft_audit_original_draw_no_longer_covers_selection")
    if (current.get("dft_audit_candidate_ids") or []) not in (replaced_ids, restored):
        raise ValueError("dft_audit_current_draw_changed_after_rejection")
    for cid in dict.fromkeys([*targets, *restored]):
        row = now.get(cid) or {}
        if not row or row != original.get(cid):
            raise ValueError(f"dft_audit_candidate_evidence_changed:{cid}")
        try:
            digest = sha256(Path(row["structure_path"]).read_bytes()).hexdigest()
        except (KeyError, TypeError, OSError) as error:
            raise ValueError(f"dft_audit_candidate_frame_unavailable:{cid}") from error
        if digest != row.get("structure_sha256"):
            raise ValueError(f"dft_audit_candidate_frame_changed:{cid}")
    return restored


def recover(state_path, session, *, apply=False):
    """Check or restore one proven draw. Caller holds the lifecycle and team locks."""
    proof_event = [
        row["proof"]
        for row in session.get("events") or []
        if row.get("kind") == "verified_validation_rejection"
        and row.get("revision") == session.get("revision")
    ]
    if (
        session.get("status") != "validation_rejected"
        or (session.get("submission") or {}).get("reason")
        != "DFT 方案未通过筛选/预算校验，需修订后再批准：['independent_dft_audit_required']"
        or len(proof_event) != 1
    ):
        raise ValueError("dft_audit_recovery_requires_verified_rejection")
    proof = proof_event[0]
    checked, frame = rejection_checkpoint(
        state_path,
        session,
        include_frame=True,
        **{key: proof[key] for key in ("checkpoint_id", "request_id", "checkpoint_sha256")},
    )
    if checked != proof:
        raise ValueError("dft_audit_rejection_proof_changed")
    path = Path(state_path)
    identity = read_identity(path, strong=True)
    current = None if apply else open_indexed_state(path)
    if current is None:
        current = read_json(path, {}) or {}
    if current.get("pending_execution_policies"):
        raise ValueError("dft_audit_recovery_pending_approval")
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        serde = CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), conn, Lock())
        before = _fields(
            serde,
            frame.get("loaded_state") or {},
            {
                "dft_audit_candidate_ids",
                "qbc_candidates",
                "confirmed_config_version",
                "active_model_version",
                "phase_diagrams",
                "phase_diagram_state",
                "model_refresh",
                "phase_diagrams_by_model",
                "extrapolation_recovery_cycle",
            },
        )
        after = _fields(
            serde, (frame.get("result") or {}).get("state") or {}, {"dft_audit_candidate_ids"}
        )
    action = session["draft"]["action"]
    restored = _restore_ids(before, after, current, action)
    if read_identity(path, strong=True) != identity:
        raise ValueError("dft_audit_recovery_state_changed")
    status = "verified" if not apply else "already_restored"
    if apply and current.get("dft_audit_candidate_ids") != restored:
        current["dft_audit_candidate_ids"] = restored
        current["dft_audit_restoration"] = {
            "checkpoint_id": proof["checkpoint_id"],
            "checkpoint_sha256": proof["checkpoint_sha256"],
            "team_id": session["team_id"],
            "revision": session["revision"],
            "replaced_ids": after["dft_audit_candidate_ids"],
            "restored_ids": restored,
        }
        _save_runner_state(current, path, expected_identity=identity)
        status = "restored"
    return {
        "status": status,
        "team_id": session["team_id"],
        "revision": session["revision"],
        "restored_audit_ids": restored,
        "matched_target_ids": [cid for cid in restored if cid in action["target_ids"]],
        "target_count": len(action["target_ids"]),
        "tasks_created": 0,
        "submitted": False,
    }
