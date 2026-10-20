"""A blocked historical MC proposal must not trap a new branch-plan request."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.graphs.dialogue.saved_control import route_saved_control
from phase_agent.graphs.dialogue.state import TurnRuntime
from phase_agent.runtime.review_requests import pending_reviews
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from tests.test_new_model_generation_gate import switched
from tests.test_region_sampling import action as region_action
from tests.test_saved_work_semantics import setup


def blocked_state():
    state = switched()
    state["tasks"] = [
        {"task_id": "old-mc", "stage": "deep_search", "status": "completed", "model_version": "old"}
    ]
    state["pending_execution_policies"] = {
        "old-plan": {
            "record_id": "old-plan",
            "agent_proposal": {
                "recommended_action": "allocate_mc_bohb",
                "raw_action": {
                    "tool": "allocate_mc_bohb",
                    "target_ids": ["old_branch"],
                    "parameters": {},
                    "task_key": "old-plan",
                    "reason": "historical plan",
                    "budget": 0,
                },
            },
        }
    }
    return state


def test_saved_semantic_request_knows_generation_is_creation_not_existing_ids(
    tmp_path, monkeypatch
):
    handler, client, context, runtime = setup(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "方案仍需重新准备。"}
    )
    facts = blocked_state()
    route_saved_control(
        {"facts": facts, "message": "生成branch方案"}, runtime, pending_reviews(facts)
    )
    payload = client.call_args.args[0]
    workflow = payload["conversation_facts"]["workflow_prerequisites"]
    assert workflow["new_model_generation"]["ready"] is False
    assert workflow["branch_generation_requires_existing_ids"] is False
    assert (
        "new_model_branch_generation_required"
        in payload["conversation_facts"]["pending"][0]["entry_errors"]
    )
    assert len(__import__("json").dumps(payload, ensure_ascii=False)) <= 8000
    assert "allowed_tools" not in payload
    assert context.scientific_call is None


def test_semantic_advance_revises_blocked_plan_without_executing_it(tmp_path, monkeypatch):
    handler, client, context, runtime = setup(tmp_path, monkeypatch, {"kind": "advance"})
    handler.review_pending = Mock(return_value={"result": {"status": "awaiting_approval"}})
    facts = blocked_state()
    before = deepcopy(facts)
    rows = pending_reviews(facts)
    routed = route_saved_control(
        {"facts": facts, "message": "先给我新一轮的生成安排吧"}, runtime, rows
    )
    assert routed["operation"] == "scientific_workflow"
    handler.review_pending.assert_not_called()
    assert context.scientific_call()["status"] == "awaiting_approval"
    handler.review_pending.assert_called_once_with(
        "old-plan",
        "modify",
        expected_state_version=rows[0]["state_version"],
        expected_proposal_hash=rows[0]["proposal_hash"],
        comment="先给我新一轮的生成安排吧",
    )
    assert facts == before and client.call_count == 1


def test_explicit_continue_prepares_revised_proposal_approval_stays_review(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import route_request

    handler, client, context, runtime = setup(tmp_path, monkeypatch, {})
    handler.review_pending = Mock(return_value={"result": {"status": "awaiting_approval"}})
    facts = blocked_state()
    routed = route_request({"facts": facts, "message": "继续"}, runtime)
    assert routed["operation"] == "scientific_workflow"
    client.assert_not_called()
    context.scientific_call()
    assert handler.review_pending.call_args.args == ("old-plan", "modify")
    handler.review_pending.reset_mock()
    assert route_request({"facts": facts, "message": "同意"}, runtime)["operation"] == "review"
    handler.review_pending.assert_not_called()


def test_generation_blocker_does_not_claim_missing_execution_receipt():
    text = format_turn_reply("原 MC 方案不能跳过新branch生成步骤。", blocked_state())
    assert "下一步：提出新的branch生成方案供审批" in text
    assert "先核对原任务与执行回执" not in text


def test_showing_blocked_plan_never_starts_revision(tmp_path, monkeypatch):
    handler, client, context, runtime = setup(tmp_path, monkeypatch, {"kind": "show_plan"})
    facts = blocked_state()
    result = route_saved_control(
        {"facts": facts, "message": "先让我看看旧方案"}, runtime, pending_reviews(facts)
    )
    assert "暂不可批准" in result["reply"] and context.scientific_call is None


def test_generation_already_done_does_not_force_another_generation(tmp_path, monkeypatch):
    from tests.test_new_model_generation_gate import generated

    handler, client, context, runtime = setup(tmp_path, monkeypatch, {"kind": "advance"})
    facts = generated()
    facts["pending_execution_policies"] = blocked_state()["pending_execution_policies"]
    result = route_saved_control(
        {"facts": facts, "message": "接着处理已有方案"}, runtime, pending_reviews(facts)
    )
    assert "reply" in result and context.scientific_call is None


def test_branch_revision_reaches_existing_online_generation_contract(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from tests.test_event_loop import _session
    from tests.test_region_sampling import Manager, H
    from pymatgen.core import Lattice, Structure
    import phase_agent.science.structures.boundary_utils as boundary_utils

    state = blocked_state()
    state["workspace_layout"] = {"workspace_root": str(tmp_path)}
    before = deepcopy(state)
    calls = []

    def client(payload):
        calls.append(payload)
        assert payload["mode"] == "autonomous_search"
        assert set(payload["allowed_tools"]) <= {"generate_branches", "pause_search"}
        assert (
            payload["decision_context"]["branch_conditions"]["new_model_generation"]["ready"]
            is False
        )
        assert payload["decision_context"]["branch_count_policy"]["count_is_result"] is True
        action = region_action()
        refs = list(
            (payload["decision_context"].get("evidence_catalog") or {}).get("references") or {}
        )[:1]
        action["branch_pool_review"] = {
            "choice": "supplement",
            "sufficiency_for_next_search": "旧池只有旧轮次证据，需要新模型新增探索",
            "rationale": "先生成再单独审阅搜索输入",
            "alternative_reason": "当前不允许旧MC跳过新增生成",
            "missing_evidence": [],
            "revisit_conditions": ["生成后审阅"],
            "evidence_refs": refs,
        }
        return action

    manager = Manager()
    manager.boundary = {"P": {}, "H": {}}
    reference = Structure(
        Lattice.cubic(5),
        ["Na", "Fe", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
    )
    monkeypatch.setattr(boundary_utils, "allowed_phases", lambda value: {"O3"})
    monkeypatch.setattr(boundary_utils, "allowed_phases_at_x", lambda p, x: {"O3"})
    monkeypatch.setattr(boundary_utils, "allowed_supercells", lambda h, p: [H])
    forbidden = Mock(side_effect=AssertionError("A proposal cannot execute a scientific handler"))
    result = run_tool_step(
        state,
        _session(),
        registry={
            name: {"handler": forbidden}
            for name in ("generate_branches", "allocate_mc_bohb", "pause_search")
        },
        agent_client=client,
        invocation_id="old-plan",
        execution_mode="interactive",
        human_feedback={"decision": "comment", "comment": "生成branch方案"},
        context={
            "unified_dialogue": True,
            "user_message": "生成branch方案",
            "manager": manager,
            "phase_references": {"O3": reference},
        },
    )
    assert result["status"] == "awaiting_approval", result.get("reason")
    assert result["agent_proposal"]["raw_action"]["tool"] == "generate_branches"
    revised = result["state"]["pending_execution_policies"]["old-plan"]
    assert revised["revision"] == 1 and len(calls) == 1
    assert (
        revised["feedback_history"][0]["prior_proposal"]
        == before["pending_execution_policies"]["old-plan"]["agent_proposal"]
    )
    assert result["state"]["tasks"] == before["tasks"] and state == before
    forbidden.assert_not_called()
