"""Approval lifetime invariants across queries, restart and interrupted preparation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest

from phase_agent.graphs.dialogue.commands import _drop_finished_pending
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
from tests.test_refresh_approval_continuity import setup
from tests.test_unified_react_dialogue import make_handler


@pytest.mark.parametrize("query", ["现在什么状态", "查看方案"])
@pytest.mark.parametrize("restarted", [False, True])
def test_query_preserves_pending_before_recovery_and_history_cleanup(
    tmp_path, monkeypatch, query, restarted
):
    forbidden = Mock(side_effect=AssertionError("Query entered recovery or execution"))
    handler = make_handler(tmp_path, forbidden)
    proposal = {
        "recommended_action": "check_convergence",
        "raw_action": {
            "tool": "check_convergence",
            "action_type": "check_convergence",
            "task_key": "p",
            "budget": 0,
            "parameters": {},
        },
    }
    state = {
        "active_model_version": "m1",
        "tasks": [],
        "pending_execution_policies": {"p": {"record_id": "p", "agent_proposal": proposal}},
        "recovery_acknowledgement": {"fingerprint": "historical", "source_stages": []},
        "post_recovery_next_step_pending": True,
    }
    write_json(handler.state_path, state)
    if restarted:
        handler = make_handler(tmp_path, forbidden)
    before = handler.state_path.read_bytes()
    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recovery_candidate", forbidden
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recover_saved_results", forbidden
    )
    reply = handler([{"role": "user", "content": query}], conversation_id="query")
    assert reply
    assert handler.state_path.read_bytes() == before
    assert (
        read_json(handler.state_path, {})["pending_execution_policies"]["p"]["agent_proposal"]
        == proposal
    )
    forbidden.assert_not_called()


@pytest.mark.parametrize("legacy_without_gate", [False, True])
def test_viewed_old_stage_plan_is_not_invited_or_approved(tmp_path, legacy_without_gate):
    from threading import RLock
    from phase_agent.runtime.plan_queries import pending_plan_reply
    from phase_agent.runtime.chat_review import review_pending
    from phase_agent.runtime.review_requests import pending_reviews

    pending = {
        "agent_proposal": {
            "recommended_action": "check_convergence",
            "raw_action": {
                "tool": "check_convergence",
                "task_key": "p",
                "parameters": {},
                "budget": 0,
            },
        }
    }
    state = {
        "active_model_version": "m1",
        "active_model": {"version": "m1"},
        "tasks": [],
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": ["deep_search"]},
        "pending_execution_policies": {"p": pending},
    }
    if legacy_without_gate:
        state["post_recovery_next_step_pending"] = False
        pending["agent_proposal"]["recommended_action"] = "prepare_local_batch_files"
        pending["agent_proposal"]["raw_action"]["tool"] = "prepare_local_batch_files"
        pending["agent_proposal"]["raw_action"]["parameters"] = {"mode": "relax_inputs"}
    before = deepcopy(state)
    text = pending_plan_reply(state)
    assert "暂不可批准" in text and "下一步：" in text and "是否同意" not in text
    assert state == before
    path = tmp_path / "state.json"
    write_json(path, state)
    handler = SimpleNamespace(lock=RLock(), state_path=path, workflow_kwargs={}, _run=Mock())
    row = pending_reviews(state)[0]
    with pytest.raises(ValueError, match="proposal_context_changed"):
        review_pending(
            handler,
            "p",
            "approve",
            expected_state_version=row["state_version"],
            expected_proposal_hash=row["proposal_hash"],
            is_sensitive=lambda p: False,
        )
    handler._run.assert_not_called()
    assert read_json(path, {}) == before


def proposal(count):
    return {
        "recommended_action": "generate_branches",
        "reason": "coverage",
        "raw_action": {
            "tool": "generate_branches",
            "task_key": "p",
            "parameters": {"total_quota": count},
            "budget": 0,
        },
    }


@pytest.mark.parametrize("latest_live_record", [False, True])
def test_cleanup_does_not_delete_new_revision_because_of_old_receipt(latest_live_record):
    old, current = proposal(100), proposal(200)
    state = {
        "pending_execution_policies": {
            "p": {"record_id": "p", "agent_proposal": current, "revision": 1}
        },
        "invocations": {"p": {"status": "failed", "agent_proposal": old}},
        "action_records": [
            {
                "record_id": "p",
                "status": "awaiting_approval" if latest_live_record else "failed",
                "agent_proposal": current if latest_live_record else old,
            }
        ],
    }
    before = deepcopy(state)
    updated, changed = _drop_finished_pending(state)
    assert not changed and updated == before and state == before


def test_exact_finished_approval_is_cleaned_but_not_live_other_object():
    state = {
        "pending_execution_policies": {
            "p": {"record_id": "p", "agent_proposal": proposal(100)},
            "other": {"record_id": "other", "agent_proposal": proposal(200)},
        },
        "invocations": {"p": {"status": "completed", "agent_proposal": proposal(100)}},
        "action_records": [],
    }
    updated, changed = _drop_finished_pending(state)
    assert changed and set(updated["pending_execution_policies"]) == {"other"}
    assert set(state["pending_execution_policies"]) == {"p", "other"}


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("known_receipt", [True, False])
def test_old_terminal_receipt_cannot_replay_as_a_new_revision(decision, known_receipt):
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from tests.test_event_loop import _session

    receipt = {"status": "failed", "reason": "old failure"}
    if known_receipt:
        receipt["agent_proposal"] = proposal(100)
    state = {
        "tasks": [],
        "pending_execution_policies": {
            "p": {"record_id": "p", "revision": 1, "agent_proposal": proposal(200)}
        },
        "invocations": {"p": deepcopy(receipt)},
    }
    tool = Mock(side_effect=AssertionError("Terminal receipt conflict executed"))
    result = run_tool_step(
        state,
        _session(),
        registry={"generate_branches": {"handler": tool}},
        invocation_id="p",
        execution_mode="interactive",
        human_feedback={
            "decision": decision,
            "comment": "同意" if decision == "approve" else "拒绝",
        },
    )
    assert result["status"] == (
        "proposal_context_changed" if decision == "approve" else "rejected_by_user"
    )
    assert bool(result["state"]["pending_execution_policies"]) == (decision == "approve")
    assert result["state"]["invocations"]["p"] == receipt
    tool.assert_not_called()


def test_cleanup_updates_the_same_turn_facts_as_persisted_state(tmp_path):
    from phase_agent.graphs.dialogue.maintenance.finetune import handle

    state = {
        "pending_execution_policies": {"p": {"record_id": "p"}},
        "action_records": [{"record_id": "p", "status": "rejected"}],
    }
    handler = SimpleNamespace(state_path=tmp_path / "state.json")
    write_json(handler.state_path, state)
    assert handle(handler, "继续", state, [], "a") is None
    assert state == read_json(handler.state_path, {})
    assert not state["pending_execution_policies"]


def blocked_authorization(tmp_path, *, native=False):
    state, session, context, registry = setup(tmp_path)
    if native:
        context["state_path"] = str(tmp_path / "state.json")
        state["confirmed_config_version"] = "c"
        state["confirmed_config"] = deepcopy(session["confirmed_snapshot"]["config"])
        write_json(context["state_path"], state)
    limit = context["effective_config"]["budgets"]["total_relative_cost"]
    state["budget_reservations"] = {
        "other": {"stage": "relax_and_feature", "status": "reserved", "reserved_cost": limit}
    }
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    blocked = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert blocked["status"] == "budget_exhausted"
    restored = json.loads(json.dumps(blocked["state"]))
    restored["budget_reservations"]["other"]["status"] = "released"
    return restored, session, context, registry


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("decision", ["reject", "comment", "modify"])
def test_negative_or_revision_feedback_never_becomes_saved_approval(tmp_path, native, decision):
    state, session, context, registry = blocked_authorization(tmp_path, native=native)
    actual_handler = registry["prepare_local_batch_files"]["handler"]
    registry["prepare_local_batch_files"]["handler"] = Mock(
        side_effect=AssertionError("Rejected or unapproved revision executed")
    )
    result = run_model_refresh_gate(
        state,
        session,
        registry=registry,
        context=context,
        human_feedback={"decision": decision, "comment": "不要执行，重新核对范围"},
    )
    assert result["status"] in {"rejected_by_user", "confirmation_required"}
    assert not result["state"]["tasks"]
    registry["prepare_local_batch_files"]["handler"].assert_not_called()
    restored = json.loads(json.dumps(result["state"]))
    proposed = run_model_refresh_gate(restored, session, registry=registry, context=context)
    assert proposed["status"] == "awaiting_approval"
    assert not proposed["state"]["tasks"]
    assert (
        proposed["action"]["task_key"]
        != state["model_refresh"]["input_authorization"]["approved_by"]
    )
    registry["prepare_local_batch_files"]["handler"].assert_not_called()
    if decision != "reject":
        assert "当前确认配置" in proposed["agent_proposal"]["reason"]
    registry["prepare_local_batch_files"]["handler"] = actual_handler
    executed = run_model_refresh_gate(
        proposed["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert executed["status"] == "prepared", executed
    assert len(executed["state"]["tasks"]) == 1


def test_existing_frozen_approval_resumes_without_new_feedback_once(tmp_path):
    state, session, context, registry = blocked_authorization(tmp_path, native=True)
    result = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert result["status"] == "prepared"
    ids = [row["task_id"] for row in result["state"]["tasks"]]
    again = run_model_refresh_gate(result["state"], session, registry=registry, context=context)
    assert [row["task_id"] for row in again["state"]["tasks"]] == ids


def test_chat_can_withdraw_an_approved_but_budget_blocked_preparation(tmp_path):
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler

    state, session, context, registry = blocked_authorization(tmp_path, native=True)
    path = tmp_path / "state.json"
    write_json(path, state)
    forbidden = Mock(side_effect=AssertionError("Withdrawal must not enter science or model"))
    handler = RunWorkflowChatHandler({"state_path": str(path), "agent_client": forbidden})
    handler._run = forbidden
    reply = handler([{"role": "user", "content": "拒绝"}], conversation_id="cancel")
    saved = read_json(path, {})
    assert saved["model_refresh"]["input_authorization"]["status"] == "revoked"
    assert "撤销" in reply and "下一步：" in reply
    assert "方案已批准" not in reply and "待恢复" not in reply
    assert not saved["tasks"]
    forbidden.assert_not_called()


@pytest.mark.parametrize("native", [False, True])
def test_fresh_rejection_does_not_poison_next_approval(tmp_path, native):
    state, session, context, registry = setup(tmp_path)
    if native:
        context["state_path"] = str(tmp_path / "state.json")
        state["confirmed_config_version"] = "c"
        state["confirmed_config"] = deepcopy(session["confirmed_snapshot"]["config"])
        write_json(context["state_path"], state)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    rejected = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="reject"
    )
    assert rejected["status"] == "rejected_by_user" and not rejected["state"]["tasks"]
    assert "下一步：" in rejected.get("reason", "")
    next_plan = run_model_refresh_gate(
        json.loads(json.dumps(rejected["state"])), session, registry=registry, context=context
    )
    assert next_plan["status"] == "awaiting_approval", next_plan
    assert next_plan["action"]["task_key"] != first["action"]["task_key"]
    prepared = run_model_refresh_gate(
        next_plan["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert prepared["status"] == "prepared", prepared
    assert len(prepared["state"]["tasks"]) == 1
