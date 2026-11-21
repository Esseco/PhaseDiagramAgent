"""Prove a frozen proposal stopped at the old-input wait without execution."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.persistence.compact_checkpoints import CompressedSerde
from phase_agent.persistence.sqlite_paths import readonly_uri
from phase_agent.tools.step_runner.file_protocol import read_json


def waiting_checkpoint(state_path, session, *, checkpoint_id, request_id, checkpoint_sha256):
    path = Path(state_path).resolve()
    action = (session.get("draft") or {}).get("action") or {}
    frozen_dft = (
        action.get("tool") == "select_dft_candidates"
        and action.get("operation") == "select_dft_candidates"
    )
    params = action.get("parameters") or {}
    frozen_training = (
        action.get("tool") == "update_mlip"
        and action.get("operation") == "prepare_mlip_training"
        and params.get("action", "RETRAIN_MLIP") == "RETRAIN_MLIP"
        and params.get("prepare_inputs_only") is True
        and bool(action.get("_simulation_plan"))
    )
    if (
        (session.get("submission") or {}).get("status") != "awaiting_manual_submission"
        or not (frozen_dft or frozen_training)
        or session.get("dispatch_request_id") not in {None, request_id}
    ):
        raise ValueError("manual_wait_not_frozen_dft")
    with closing(sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
    if not row or row[1] != checkpoint_id or hashlib.sha256(row[3]).hexdigest() != checkpoint_sha256:
        raise ValueError("manual_wait_checkpoint_changed")
    payload = CompressedSerde(JsonPlusSerializer()).loads_typed(
        (row[2].removeprefix("phase-cas-v1:"), row[3])
    )
    values = payload.get("channel_values") or {}
    response = values.get("completed_response") or {}
    frame = values.get("business_frame") or {}
    wait = response.get("manual_wait") or {}
    ids = wait.get("task_ids") or []
    if (
        values.get("request_id") != request_id
        or values.get("phase") != "wait_checked"
        or values.get("current_node") != "results_wait_gate"
        or values.get("node_status") != "waiting"
        or not frame.get("runtime_state_path")
        or Path(frame["runtime_state_path"]).resolve() != path
        or response.get("status") != "awaiting_manual_submission"
        or response.get("steps_executed") != 0
        or response.get("submitted") is not False
        or any(response.get(k) for k in ("record_id", "agent_proposal", "execution", "execution_result", "events"))
        or not ids or len(ids) != len(set(ids))
        or wait.get("waiting_task_count") != len(ids)
        or wait.get("waiting_by_stage") != {"dft_single_point": len(ids)}
    ):
        raise ValueError("manual_wait_checkpoint_does_not_prove_zero_execution")
    state = read_json(path, {}) or {}
    tasks = state.get("tasks") or []
    by_id = {item.get("task_id"): item for item in tasks}
    if (
        len(by_id) != len(tasks)
        or state.get("pending_execution_policies")
        or any(
            (by_id.get(tid) or {}).get("stage") != "dft_single_point"
            or (by_id.get(tid) or {}).get("status") != "pending"
            or not (by_id.get(tid) or {}).get("input_path")
            for tid in ids
        )
    ):
        raise ValueError("manual_wait_tasks_changed")
    receipt_path = path.parent / "execution_receipts.sqlite"
    if receipt_path.exists():
        with closing(sqlite3.connect(readonly_uri(receipt_path), uri=True)) as conn:
            if any(
                json.loads(identity).get("invocation_id") == request_id
                for (identity,) in conn.execute("SELECT identity FROM receipts")
            ):
                raise ValueError("manual_wait_has_execution_receipt")
    return {"checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint_sha256,
            "request_id": request_id, "thread_id": row[0],
            "classification": "manual_wait_before_dispatch", "executed": False}
