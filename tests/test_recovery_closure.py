from copy import deepcopy
from types import SimpleNamespace
from threading import RLock
import sqlite3
from unittest.mock import Mock, patch

import pytest

from phase_agent.tools.state.execution_receipts import (
    begin_execution,
    execution_events,
    recovery_report,
    record_execution_return,
    record_business_saved,
)
from phase_agent.tools.state.execution_reconciliation import (
    propose_reconciliation,
    apply_reconciliation,
)
from phase_agent.tools.step_runner.file_protocol import write_json, read_json

IDENTITY = {
    "invocation_id": "old",
    "config_version": "c",
    "action_hash": "h",
    "tool": "generate_branches",
}
ATTESTATION = {
    "evidence_note": "已核对输出目录、登记台账与作业来源，无科学计算产出",
    "external_jobs": "none_found",
    "output_inventory": "checked_no_outputs",
    "external_cost": 0.0,
}


def interrupted(tmp_path):
    path = tmp_path / "state.json"
    begin_execution(path, IDENTITY, action={"tool": "generate_branches", "task_key": "key"})
    state = {
        "active_model_version": "old-model",
        "action_records": [
            {"record_id": "old", "status": "failed", "final_action": {"task_key": "key"}}
        ],
        "tasks": [],
        "budget_reservations": {
            "key": {"status": "reserved", "stage": "generation", "reserved_cost": 20}
        },
        "reserved_relative_cost": 20,
        "effective_decisions": {
            "key": {
                "status": "reconciliation_required",
                "tool": "generate_branches",
                "record_id": "old",
            }
        },
        "pending_execution_policies": {
            "old": {"agent_proposal": {"raw_action": {"task_key": "key"}}}
        },
    }
    return path, state


def test_no_effect_requires_review_releases_only_matching_reservation_and_keeps_receipt(tmp_path):
    path, state = interrupted(tmp_path)
    state["budget_reservations"]["other"] = {"status": "reserved", "reserved_cost": 5}
    state["reserved_relative_cost"] += 5
    before = deepcopy(state)
    proposed = propose_reconciliation(path, state, "old", "verified_no_effect", ATTESTATION)
    assert state == before
    assert recovery_report(path, proposed["state"])["status"] == "reconciliation_required"
    with pytest.raises(ValueError, match="confirm_sensitive"):
        apply_reconciliation(path, proposed["state"], proposed["plan_id"], "approve")
    receipt = (tmp_path / "execution_receipts.sqlite").read_bytes()
    result = apply_reconciliation(path, proposed["state"], proposed["plan_id"], "confirm_sensitive")
    assert result["status"] == "reconciled_no_effect"
    assert recovery_report(path, result["state"])["status"] == "clear"
    assert result["state"]["reserved_relative_cost"] == 5
    assert result["state"]["budget_reservations"]["other"] == before["budget_reservations"]["other"]
    assert result["state"]["budget_usage"]["total_relative_cost"] == 0
    assert result["state"]["active_model_version"] == "old-model"
    assert not result["state"]["pending_execution_policies"]
    assert (tmp_path / "execution_receipts.sqlite").read_bytes() == receipt
    assert not begin_execution(path, IDENTITY)["allowed"]
    assert not begin_execution(path, {**IDENTITY, "action_hash": "changed"})["allowed"]


@pytest.mark.parametrize(
    "change",
    [
        {"external_jobs": "unknown"},
        {"evidence_note": ""},
        {"external_cost": 1.0},
        {"external_cost": True},
        {"output_inventory": "registered_outputs"},
    ],
)
def test_incomplete_attestations_cannot_end_old_action(tmp_path, change):
    path, state = interrupted(tmp_path)
    with pytest.raises(ValueError):
        propose_reconciliation(path, state, "old", "verified_no_effect", {**ATTESTATION, **change})


def test_registered_tasks_can_be_reused_but_never_declared_no_effect(tmp_path):
    path, state = interrupted(tmp_path)
    artifact = tmp_path / "result.json"
    artifact.write_text('{"energy": 1}')
    state["tasks"] = [
        {"task_id": "t", "task_key": "key", "status": "completed", "result_path": str(artifact)}
    ]
    with pytest.raises(ValueError, match="已登记任务"):
        propose_reconciliation(path, state, "old", "verified_no_effect", ATTESTATION)
    original = artifact.read_bytes()
    attestation = {
        **ATTESTATION,
        "external_jobs": "registered_only",
        "output_inventory": "registered_outputs",
        "external_cost": None,
    }
    plan = propose_reconciliation(path, state, "old", "verified_registered_effects", attestation)
    result = apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive")
    assert result["status"] == "reconciled_existing_effects"
    assert result["state"]["tasks"] == state["tasks"]
    assert result["state"]["reserved_relative_cost"] == 20
    assert artifact.read_bytes() == original
    assert recovery_report(path, result["state"])["status"] == "clear"
    assert not begin_execution(path, IDENTITY)["allowed"]


def test_changed_file_or_business_state_cannot_use_old_review(tmp_path):
    path, state = interrupted(tmp_path)
    artifact = tmp_path / "result.json"
    artifact.write_text("one")
    state["tasks"] = [
        {"task_id": "t", "task_key": "key", "status": "completed", "result_path": str(artifact)}
    ]
    plan = propose_reconciliation(
        path,
        state,
        "old",
        "verified_registered_effects",
        {**ATTESTATION, "output_inventory": "registered_outputs"},
    )
    artifact.write_text("two")
    with pytest.raises(ValueError, match="evidence_changed"):
        apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive")
    changed = deepcopy(plan["state"])
    changed["active_model_version"] = "new-model"
    with pytest.raises(ValueError, match="state_changed"):
        apply_reconciliation(path, changed, plan["plan_id"], "confirm_sensitive")


def test_rejection_retains_block_and_next_proposal_has_new_identity(tmp_path):
    path, state = interrupted(tmp_path)
    plan = propose_reconciliation(path, state, "old", "verified_no_effect", ATTESTATION)
    rejected = apply_reconciliation(path, plan["state"], plan["plan_id"], "reject")
    assert recovery_report(path, rejected["state"])["unsettled"]
    assert rejected["state"]["reserved_relative_cost"] == 20
    again = propose_reconciliation(
        path, rejected["state"], "old", "verified_no_effect", ATTESTATION
    )
    assert again["plan_id"] != plan["plan_id"]


def test_duplicate_propose_is_idempotent_and_revised_evidence_changes_hash(tmp_path):
    from phase_agent.runtime.review_requests import pending_reviews

    path, state = interrupted(tmp_path)
    first = propose_reconciliation(path, state, "old", "verified_no_effect", ATTESTATION)
    again = propose_reconciliation(path, first["state"], "old", "verified_no_effect", ATTESTATION)
    assert first == again
    revised = propose_reconciliation(
        path,
        again["state"],
        "old",
        "verified_no_effect",
        {**ATTESTATION, "evidence_note": "补充核对全部输出目录"},
    )
    rows = [
        r for r in pending_reviews(revised["state"]) if r.get("review_kind") == "execution_recovery"
    ]
    assert len(rows) == 1 and rows[0]["revision"] == 1 and rows[0]["review_card"]["sensitive"]


def test_tool_return_is_journaled_before_business_save_and_never_replayed(tmp_path):
    from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
    from phase_agent.graphs.execution_recovery_graph import execution_recovery_report

    path = tmp_path / "state.json"
    artifact = tmp_path / "prepared.json"
    calls = []

    def handler(**kwargs):
        calls.append(1)
        artifact.write_text("prepared")
        return {
            "status": "prepared",
            "result_path": str(artifact),
            "state": {"large_state": "not journaled"},
        }

    registry = {"generate_branches": {"handler": handler}}
    context = {"state_path": str(path), "invocation_id": "old", "config_version": "c"}
    action = {"tool": "generate_branches", "task_key": "key"}
    execution = execute_tool_action(action, registry=registry, context=context)
    events = execution_events(path, "old")
    assert [e["event"] for e in events] == ["claimed", "tool_returned"]
    assert "large_state" not in str(events)
    report = execution_recovery_report(path, {})
    assert report["checks"][0]["artifact_checks"][0]["sha256"]
    assert report["unsettled"]
    assert (
        execute_tool_action(action, registry=registry, context=context)["status"]
        == "execution_reconciliation_required"
    )
    assert calls == [1]
    business = {"invocations": {"old": {"status": "completed", "execution": execution}}}
    write_json(path, business)
    record_business_saved(path, business)
    assert [e["event"] for e in execution_events(path, "old")] == [
        "claimed",
        "tool_returned",
        "business_saved",
    ]
    record_business_saved(path, business)
    assert len(execution_events(path, "old")) == 3
    assert recovery_report(path, business)["status"] == "clear"


def test_return_outcome_and_evidence_commit_together(tmp_path):
    path = tmp_path / "state.json"
    begin_execution(path, IDENTITY)
    with pytest.raises(TypeError):
        record_execution_return(path, IDENTITY, "completed", evidence={"invalid": object()})
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as connection:
        assert connection.execute("SELECT phase,outcome FROM receipts").fetchone() == (
            "started",
            None,
        )
    assert [e["event"] for e in execution_events(path, "old")] == ["claimed"]


def test_old_database_readonly_inspection_does_not_upgrade(tmp_path):
    path = tmp_path / "execution_receipts.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE receipts(action_id TEXT PRIMARY KEY,identity TEXT,phase TEXT,outcome TEXT)"
        )
    before = path.read_bytes()
    assert execution_events(tmp_path / "state.json", "old") == []
    assert recovery_report(tmp_path / "state.json", {})["status"] == "clear"
    assert before == path.read_bytes()


def test_control_queue_review_and_retry_do_not_call_workflow(tmp_path):
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.runtime.chat_review import review_pending

    path, state = interrupted(tmp_path)
    write_json(path, state)
    handler = SimpleNamespace(state_path=path, lock=RLock(), workflow_kwargs={}, _run=Mock())
    handler.review_pending = lambda *args, **kwargs: review_pending(handler, *args, **kwargs)
    control = LocalAgentControl(handler)
    result = control.propose_recovery(
        {"invocation_id": "old", "resolution": "verified_no_effect", **ATTESTATION}
    )
    row = next(r for r in control.pending()["pending"] if r["plan_id"] == result["plan_id"])
    args = {
        "plan_id": row["plan_id"],
        "expected_state_version": row["state_version"],
        "expected_proposal_hash": row["proposal_hash"],
    }
    assert control.decide("confirm_sensitive", **args)["status"] == "reconciled_no_effect"
    assert control.decide("confirm_sensitive", **args)["status"] == "already_processed"
    handler._run.assert_not_called()
    assert recovery_report(path, read_json(path))["status"] == "clear"


def test_json_sync_failure_keeps_previous_state_and_cleans_temp(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"status": "old"})
    with patch(
        "phase_agent.tools.step_runner.file_protocol.os.fsync",
        side_effect=OSError("disk flush failed"),
    ):
        with pytest.raises(OSError, match="flush failed"):
            write_json(path, {"status": "new"})
    assert read_json(path) == {"status": "old"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_model_activation_cannot_bypass_unsettled_execution(tmp_path):
    from phase_agent.runtime.review_requests import additional_reviews, review_additional
    from tests.test_hitl_closure import activation_state

    path, interrupted_state = interrupted(tmp_path)
    state = {
        **activation_state(),
        **{key: value for key, value in interrupted_state.items() if key != "active_model_version"},
    }
    row = next(row for row in additional_reviews(state) if row["review_kind"] == "model_activation")
    handler = SimpleNamespace(state_path=path)
    with pytest.raises(ValueError, match="先核对"):
        review_additional(handler, state, row, "confirm_sensitive", "approved")


def test_business_saved_acknowledgement_rejects_different_disk_state(tmp_path):
    path = tmp_path / "state.json"
    begin_execution(path, IDENTITY)
    record_execution_return(path, IDENTITY, "completed")
    write_json(path, {"invocations": {}})
    with pytest.raises(ValueError, match="business state changed"):
        record_business_saved(
            path,
            {"invocations": {"old": {"status": "completed", "execution": {"status": "completed"}}}},
        )
    assert all(row["event"] != "business_saved" for row in execution_events(path, "old"))


def test_return_journal_keeps_related_task_refs_but_omits_unrelated_state():
    from phase_agent.tools.state.execution_return_evidence import return_evidence

    result = {
        "status": "completed",
        "result": {
            "state": {
                "secret": "exclude",
                "tasks": [
                    {
                        "task_id": "related",
                        "task_key": "key",
                        "result_path": "registered.json",
                        "large_arrays": [1, 2, 3],
                    },
                    {"task_id": "unrelated", "task_key": "other"},
                ],
            }
        },
    }
    evidence = return_evidence(result, action={"task_key": "key"})
    assert evidence["returned_task_refs"] == [
        {"task_id": "related", "task_key": "key", "result_path": "registered.json"}
    ]
    assert (
        "secret" not in str(evidence)
        and "large_arrays" not in str(evidence)
        and "unrelated" not in str(evidence)
    )


def test_returned_but_unregistered_task_cannot_be_waived_as_no_effect(tmp_path):
    path, state = interrupted(tmp_path)
    record_execution_return(
        path,
        IDENTITY,
        "completed",
        evidence={"returned_task_refs": [{"task_id": "lost", "task_key": "key"}]},
    )
    with pytest.raises(ValueError, match="完整登记"):
        propose_reconciliation(path, state, "old", "verified_no_effect", ATTESTATION)


@pytest.mark.parametrize("amount,total", [(None, 20), (True, 20), (float("nan"), 20), (20, 5)])
def test_no_effect_cannot_silently_repair_corrupt_budget(tmp_path, amount, total):
    path, state = interrupted(tmp_path)
    state["budget_reservations"]["key"]["reserved_cost"] = amount
    state["reserved_relative_cost"] = total
    with pytest.raises(ValueError, match="预算"):
        propose_reconciliation(path, state, "old", "verified_no_effect", ATTESTATION)


def test_recovery_http_endpoint_auth_proposal_and_review(tmp_path):
    import json
    from threading import Thread
    from urllib.request import Request, ProxyHandler, build_opener
    from urllib.error import HTTPError
    from phase_agent.runtime.agent_api import create_server
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.runtime.chat_review import review_pending

    path, state = interrupted(tmp_path)
    write_json(path, state)
    handler = SimpleNamespace(state_path=path, lock=RLock(), workflow_kwargs={}, _run=Mock())
    handler.review_pending = lambda *args, **kwargs: review_pending(handler, *args, **kwargs)
    server = create_server(
        lambda *args, **kwargs: "unused",
        api_key="fixture-tool-key",
        port=0,
        local_control=LocalAgentControl(handler),
        control_api_key="fixture-control-key",
    )
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    opener = build_opener(ProxyHandler({}))

    def post(suffix, body, token="fixture-control-key"):
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}" + suffix,
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        with opener.open(request, timeout=4) as response:
            return json.load(response)

    try:
        body = {"invocation_id": "old", "resolution": "verified_no_effect", **ATTESTATION}
        with pytest.raises(HTTPError) as denied:
            post("/phase/recovery/propose", body, "wrong-key")
        assert denied.value.code == 401
        with pytest.raises(HTTPError) as invalid:
            post("/phase/recovery/propose", {**body, "external_cost": "0"})
        assert invalid.value.code == 400
        proposed = post("/phase/recovery/propose", body)
        row = next(
            row
            for row in LocalAgentControl(handler).pending()["pending"]
            if row["plan_id"] == proposed["plan_id"]
        )
        reviewed = post(
            "/phase/decision",
            {
                "plan_id": row["plan_id"],
                "decision": "confirm_sensitive",
                "state_version": row["state_version"],
                "proposal_hash": row["proposal_hash"],
                "comment": "已核对原计划和预算",
            },
        )
        assert reviewed["status"] == "reconciled_no_effect"
        assert (
            read_json(path)["execution_reconciliations"]["old"]["reviewer_comment"]
            == "已核对原计划和预算"
        )
        handler._run.assert_not_called()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
