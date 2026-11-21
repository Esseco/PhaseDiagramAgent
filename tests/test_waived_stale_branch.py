import copy
import json
import sqlite3

import pytest

from phase_agent.tools.state.scheduler_terminal_evidence import user_waived_dft_reservation
from phase_agent.agents.strategic.codex_stale_submission import stale_submission_checkpoint
from tests.test_codex_stale_submission import fixture


def waiver():
    task = {"task_id": "d1", "task_key": "k", "stage": "dft_single_point",
            "status": "cancelled", "config_version": "c", "model_version": "m",
            "upload_operation_id": "batch"}
    reservation = {"stage": "dft_single_point", "status": "reserved",
                   "config_version": "c", "reserved_cost": 100}
    decision = {"decision": "close", "question_id": "q", "user_message": "不再等待",
                "asked_task_ids": ["d1"], "waived_task_ids": ["d1"],
                "cancelled_task_ids": ["d1"],
                "scope": {"model_version": "m", "upload_operation_id": "batch"}}
    return {"tasks": [task], "dft_recovery_decisions": [decision]}, reservation


def test_user_waiver_keeps_cost_unsettled_and_is_precisely_bound():
    state, reservation = waiver()
    old = copy.deepcopy(state), copy.deepcopy(reservation)
    assert user_waived_dft_reservation(state, "k", reservation)
    assert (state, reservation) == old


@pytest.mark.parametrize("case", ["missing", "wrong_task", "wrong_model", "wrong_batch",
                                  "not_cancelled", "wrong_config", "duplicate", "review"])
def test_user_waiver_rejects_incomplete_or_unrelated_evidence(case):
    state, reservation = waiver()
    task, decision = state["tasks"][0], state["dft_recovery_decisions"][0]
    if case == "missing":
        decision["user_message"] = ""
    elif case == "wrong_task":
        decision["waived_task_ids"] = ["other"]
    elif case == "wrong_model":
        decision["scope"]["model_version"] = "other"
    elif case == "wrong_batch":
        decision["scope"]["upload_operation_id"] = "other"
    elif case == "not_cancelled":
        task["status"] = "running"
    elif case == "wrong_config":
        task["config_version"] = "other"
    elif case == "duplicate":
        state["tasks"].append(copy.deepcopy(task))
    else:
        reservation["reconciliation_required"] = True
    assert not user_waived_dft_reservation(state, "k", reservation)


def branch_session(path):
    path, session, proof = fixture(path)
    session["status"] = "submission_uncertain"
    session.pop("submission")
    session["draft"]["action"] = {"tool": "generate_branches", "operation": "generate_branches",
                                   "task_key": "branch"}
    return path, session, proof


def test_branch_pre_dispatch_reconcile_requires_checkpoint(tmp_path):
    path, session, proof = branch_session(tmp_path)
    before = path.read_bytes()
    assert stale_submission_checkpoint(path, session, **proof)["executed"] is False
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="checkpoint_changed"):
        stale_submission_checkpoint(path, session, **{**proof, "checkpoint_sha256": "wrong"})


def test_branch_claim_blocks_reconcile(tmp_path):
    path, session, proof = branch_session(tmp_path)
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("INSERT INTO execution_events VALUES (?, ?, ?, ?)",
                     ("id", "claimed", json.dumps({"action": {"task_key": "branch"}}), "now"))
    with pytest.raises(ValueError, match="execution_receipt"):
        stale_submission_checkpoint(path, session, **proof)
