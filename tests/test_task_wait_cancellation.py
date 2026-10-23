from copy import deepcopy

import pytest

from phase_agent.tools.state.task_wait_cancellation import (
    cancellation_scopes,
    cancel_task_waits,
    commit_task_wait_cancellation,
    latest_cancellation_facts,
)
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.state.workflow_control import restore_control


def state_fixture():
    tasks = [
        {
            "task_id": f"MC-{i}",
            "task_key": f"mc:{i}",
            "stage": "deep_search",
            "status": "pending",
            "config_version": "c",
            "model_version": "m",
            "relax_mc_bundle_id": "bundle",
            "slurm_batch_id": "remote-3",
        }
        for i in range(5)
    ]
    return {
        "tasks": tasks,
        "pending_tasks": deepcopy(tasks),
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "budget_reservations": {"mc:0": {"status": "submitted", "reserved_cost": 7}},
        "used_relative_cost": 10,
        "reserved_relative_cost": 7,
        "processed_task_ids": [],
        "execution_reconciliations": {},
        "invocations": {},
        "pending_execution_policies": {},
        "slurm_batches": [{"batch_id": "remote-3", "job_id": "3743912"}],
    }


def test_exact_cancellation_preserves_remote_evidence_and_unknown_costs():
    state = state_fixture()
    original = deepcopy(state)
    scope = cancellation_scopes(state)[0]
    assert scope["count"] == 5
    updated, records = cancel_task_waits(
        state, scope["scope_id"], user_message="取消这五个MC，用已有结果继续"
    )
    assert state == original
    assert len(records) == 5 and updated["pending_tasks"] == []
    assert all(
        row["status"] == "cancelled" and row["recovery_cancelled"] for row in updated["tasks"]
    )
    for key in (
        "slurm_batches",
        "budget_reservations",
        "used_relative_cost",
        "reserved_relative_cost",
        "processed_task_ids",
    ):
        assert updated[key] == original[key]
    assert all(row["remote_cancel_sent"] is False for row in records.values())


def test_scope_is_not_permission_to_cancel_new_or_changed_tasks():
    state = state_fixture()
    scope = cancellation_scopes(state)[0]["scope_id"]
    state["tasks"][0]["status"] = "completed"
    with pytest.raises(ValueError, match="scope_changed"):
        cancel_task_waits(state, scope, user_message="取消")
    assert state["tasks"][0]["status"] == "completed"
    state["active_model_version"] = "new-model"
    assert cancellation_scopes(state) == []


def test_cancellation_survives_old_checkpoint_and_blocks_stale_writers(tmp_path):
    path = tmp_path / "state.json"
    old = state_fixture()
    write_json(path, old)
    scope = cancellation_scopes(old)[0]["scope_id"]
    updated = commit_task_wait_cancellation(path, old, scope, user_message="取消这五个MC")
    restored = restore_control(path, old)
    assert restored == updated
    with pytest.raises(ValueError, match="stale_workflow_control_revision"):
        write_json(path, old)
    rebound = deepcopy(old)
    rebound["tasks"][0]["task_key"] = "other-science"
    with pytest.raises(ValueError, match="binding_changed"):
        restore_control(path, rebound)
    assert read_json(path) == updated


def test_late_terminal_result_cannot_reactivate_or_settle_cancelled_mc():
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results

    state = state_fixture()
    updated, _ = cancel_task_waits(
        state, cancellation_scopes(state)[0]["scope_id"], user_message="取消"
    )
    incoming = {**state["tasks"][0], "status": "failed", "actual_cost": 2}
    reconciled = reconcile_task_results(updated, [incoming])
    assert reconciled["reconciled"][0]["status"] == "rejected"
    for key in (
        "tasks",
        "pending_tasks",
        "budget_reservations",
        "used_relative_cost",
        "reserved_relative_cost",
    ):
        assert reconciled["state"][key] == updated[key]


def test_changed_bundle_after_cancellation_needs_manual_review(tmp_path, monkeypatch):
    from phase_agent.tools.relax_mc_batch import results

    state = state_fixture()
    updated, _ = cancel_task_waits(
        state, cancellation_scopes(state)[0]["scope_id"], user_message="取消"
    )
    archive = tmp_path / "results.zip"
    archive.write_bytes(b"keep raw archive")
    updated["relax_mc_batches"] = {
        "bundle": {"model_version": "m", "config_version": "c", "import_checksum": "old"}
    }
    monkeypatch.setattr(results, "result_archive", lambda record: archive)
    monkeypatch.setattr(
        results, "import_bundle", lambda *a, **kw: pytest.fail("cancelled return imported")
    )
    context = {"event_state": updated, "effective_config": {"mlip": {}}, "config_version": "c"}
    result = results.import_results(action={"parameters": {"bundle_id": "bundle"}}, context=context)
    assert result["status"] == "manual_review_required"
    assert result["state"]["tasks"] == updated["tasks"]
    assert result["state"]["budget_reservations"] == updated["budget_reservations"]
    assert "return_diagnostic" not in updated["relax_mc_batches"]["bundle"]
    assert archive.read_bytes() == b"keep raw archive"


def test_typed_dialogue_control_cannot_smuggle_submission():
    from phase_agent.decisions.agent.dialogue_contract import dialogue_errors

    action = {
        "kind": "cancel_waits",
        "scope_id": cancellation_scopes(state_fixture())[0]["scope_id"],
    }
    assert dialogue_errors(action, True) == []
    assert dialogue_errors(action, False)
    assert dialogue_errors({**action, "tool": "run_calculation_stage"}, True)
    assert dialogue_errors({**action, "target_ids": ["other"]}, True)


def test_semantic_cancellation_then_original_planning_without_approval(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts

    state = state_fixture()
    scope = cancellation_scopes(state)[0]["scope_id"]
    calls = []

    def model(payload):
        calls.append(payload)
        assert payload["state"]["task_wait_cancellation_scopes"][0]["count"] == 5
        return {"kind": "cancel_waits", "scope_id": scope}

    handler = make_handler(tmp_path, model)
    handler.turn_context = {}
    write_json(handler.state_path, state)
    observed = []
    result = decide_from_saved_facts(
        handler,
        state,
        "取消这五个MC，用已有结果继续",
        lambda: (
            observed.append(read_json(handler.state_path)) or {"status": "scientific_requested"}
        ),
    )
    assert result["status"] == "scientific_requested"
    assert len(calls) == 1 and len(observed) == 1
    assert observed[0]["pending_tasks"] == []
    assert not observed[0].get("pending_execution_policies")


def test_ambiguous_cancel_question_is_read_only(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts

    state = state_fixture()
    handler = make_handler(
        tmp_path, lambda payload: {"kind": "answer", "answer": "请指出要停止等待的MC批次。"}
    )
    handler.turn_context = {}
    write_json(handler.state_path, state)
    result = decide_from_saved_facts(
        handler, state, "可以取消吗", lambda: pytest.fail("question entered science")
    )
    assert result["status"] == "answered" and read_json(handler.state_path) == state


def test_cancellation_intent_reaches_scientific_summary_after_restart():
    from phase_agent.analysis.state.build_decision_context import build_decision_context
    from phase_agent.decisions.agent.summary_request import summary_payload

    state = state_fixture()
    updated, _ = cancel_task_waits(
        state,
        cancellation_scopes(state)[0]["scope_id"],
        user_message="取消这五个MC，用已有结果继续",
    )
    context = build_decision_context(updated)
    facts = latest_cancellation_facts(updated)
    assert facts["task_count"] == 5 and "已有结果继续" in facts["user_message"]
    assert context["task_wait_cancellation"] == facts
    assert (
        summary_payload({"state": {}, "decision_context": context})[0]["state"][
            "task_wait_cancellation"
        ]
        == facts
    )
    updated["active_model_version"] = "new-model"
    assert latest_cancellation_facts(updated) == {}
