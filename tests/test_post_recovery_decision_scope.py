from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.graphs.dialogue.nodes import route_request


@pytest.mark.parametrize(
    "stages,expected",
    [
        (["deep_search"], {"generate_branches", "select_dft_candidates"}),
        (
            ["dft_single_point"],
            {
                "generate_branches",
                "select_dft_candidates",
                "update_mlip",
                "check_convergence",
                "pause_search",
                "adjust_strategy",
            },
        ),
    ],
)
def test_recovery_tools_follow_the_recovered_stage(stages, expected):
    from phase_agent.policies.recovery_transition import post_recovery_tools

    state = {
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": stages},
    }
    assert post_recovery_tools(state) == expected


@pytest.mark.parametrize("status", ["failed", "timeout", "pending", "completed"])
def test_recovery_retry_route_requires_an_actionable_failed_task(status):
    from phase_agent.policies.recovery_transition import post_recovery_tools

    state = {
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": ["dft_single_point"]},
        "tasks": [{"task_id": "dft-one", "stage": "dft_single_point", "status": status}],
    }
    before = deepcopy(state)
    assert ("restart_failed_task" in post_recovery_tools(state)) is (
        status in {"failed", "timeout"}
    )
    assert state == before


@pytest.mark.parametrize("retry_status", ["pending", "completed"])
def test_existing_same_scope_retry_does_not_unlock_another_retry(retry_status):
    from phase_agent.policies.recovery_transition import post_recovery_tools

    scope = {"stage": "dft_single_point", "model_version": "m1", "structure_id": "s1"}
    state = {
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": ["dft_single_point"]},
        "tasks": [
            {**scope, "task_id": "failed", "status": "failed", "retry_task_id": "retry"},
            {**scope, "task_id": "retry", "status": retry_status},
        ],
    }
    before = deepcopy(state)
    assert "restart_failed_task" not in post_recovery_tools(state)
    assert state == before


def test_legacy_recovery_marker_uses_current_dft_assessment(monkeypatch):
    from phase_agent.policies.recovery_transition import post_recovery_tools

    monkeypatch.setattr(
        "phase_agent.evidence.post_dft_assessment.post_dft_assessment",
        lambda *_: {"status": "evaluated"},
    )
    assert "update_mlip" in post_recovery_tools({"post_recovery_next_step_pending": True})


def test_current_model_registered_branches_unlock_input_review():
    from phase_agent.policies.recovery_transition import post_recovery_tools

    state = {
        "post_recovery_next_step_pending": True,
        "active_model_version": "new",
        "model_refresh": {
            "status": "completed",
            "old_model_version": "old",
            "new_model_version": "new",
        },
        "generation_history": [
            {"model_version": "new", "new_branch_ids": ["b1"], "registered_ids": ["s1"]}
        ],
    }
    assert "prepare_local_batch_files" in post_recovery_tools(state)
    state["generation_history"][0]["model_version"] = "old"
    assert "prepare_local_batch_files" not in post_recovery_tools(state)


def test_dft_recovery_does_not_supersede_a_pending_finetune_proposal(tmp_path, monkeypatch):
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    facts = {
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": ["dft_single_point"]},
        "pending_execution_policies": {
            "training-plan": {"agent_proposal": {"raw_action": {"tool": "update_mlip"}}}
        },
    }
    write_json(path, facts)
    before = deepcopy(facts)
    monkeypatch.setattr("phase_agent.runtime.review_requests.pending_reviews", lambda _: [])
    monkeypatch.setattr(
        "phase_agent.graphs.dialogue.saved_control.route_saved_control",
        lambda *_: {"reply": "原微调方案等待批准"},
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(handler=SimpleNamespace(state_path=path), conversation_id="test")
    )

    result = route_request({"message": "查看方案", "facts": facts}, runtime)

    assert result["reply"] == "原微调方案等待批准"
    assert facts == before


@pytest.mark.parametrize("stage", ["deep_search", "dft_single_point"])
def test_scientific_planner_receives_recovered_stage_routes(monkeypatch, stage):
    from phase_agent.agents.common.post_dft_review import REVIEW_TOOLS
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    monkeypatch.setattr(
        "phase_agent.evidence.post_dft_assessment.post_dft_assessment",
        lambda *_: (
            {"status": "evaluated", "scope_key": "dft-round"}
            if stage == "dft_single_point"
            else None
        ),
    )
    monkeypatch.setattr(
        "phase_agent.science.qbc.post_mc_candidates.post_mc_candidates", lambda *_: []
    )
    monkeypatch.setattr(
        "phase_agent.science.mc.second_round_state.second_round_completed", lambda *_: True
    )
    seen = []

    def propose(_, **kwargs):
        seen.append(set(kwargs["allowed_tools"]))
        return {"tool": "pause_search", "fallback_reason": "llm_failed: synthetic"}

    tools = [*REVIEW_TOOLS, "prepare_local_batch_files"]
    result = select_initial_proposal(
        current={
            "active_model_version": "m1",
            "post_recovery_next_step_pending": True,
            "recovery_acknowledgement": {"source_stages": [stage]},
        },
        mode="autonomous",
        config={"agent": {"allowed_tools": tools}},
        context={"unified_dialogue": True},
        decision_state={},
        registry={name: {"handler": lambda: None} for name in tools},
        agent_client=lambda _: None,
        invocation_id="test",
        propose=propose,
        revise=None,
        prepare_debug=None,
        is_verified_second=None,
        mc_block=None,
        model_failed=lambda _: True,
        record_id_factory=lambda *_: "record",
    )

    expected = (
        (set(REVIEW_TOOLS) - {"restart_failed_task"})
        if stage == "dft_single_point"
        else {"generate_branches", "select_dft_candidates"}
    )
    assert seen == [expected]
    assert result["status"] == "not_configured"
    assert not result["state"].get("pending_execution_policies")


@pytest.mark.parametrize("stage", ["deep_search", "dft_single_point"])
def test_dialogue_and_scientific_gate_share_the_same_scope(tmp_path, monkeypatch, stage):
    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts
    from phase_agent.policies.recovery_transition import post_recovery_tools
    from tests.test_unified_react_dialogue import make_handler

    seen = []

    def decision(_, payload):
        seen.append(set(payload["allowed_tools"]))
        return {"kind": "advance", "reason": "Review recovered evidence"}

    monkeypatch.setattr(
        "phase_agent.graphs.decision_team.backend.request_decision_action", decision
    )
    handler = make_handler(tmp_path, lambda _: None)
    state = {
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": [stage]},
    }
    forwarded = {"forwarded_to_scientific_planner": True}

    result = decide_from_saved_facts(handler, state, "分析下一步，等我批准", lambda: forwarded)

    assert seen == [post_recovery_tools(state)]
    assert result is forwarded
    assert not state.get("pending_execution_policies")


@pytest.mark.parametrize("stage", ["deep_search", "dft_single_point"])
@pytest.mark.parametrize(
    "outcome,status",
    [
        ({"kind": "answer", "answer": "已回收12个，3个失败，正在解释原因。"}, "answered"),
        ({"kind": "inspect_evidence", "topics": ["dft", "model"]}, "evidence_requested"),
        (
            {"kind": "inspect_configuration", "answer": "查看当前设置"},
            "configuration_view_requested",
        ),
        (
            {"kind": "review", "plan_id": "shown-plan", "decision": "approve"},
            "proposal_review_requested",
        ),
    ],
)
def test_post_recovery_scope_does_not_override_dialogue_kinds(
    tmp_path, monkeypatch, stage, outcome, status
):
    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts
    from tests.test_unified_react_dialogue import make_handler

    monkeypatch.setattr(
        "phase_agent.graphs.decision_team.backend.request_decision_action",
        lambda *_: deepcopy(outcome),
    )
    handler = make_handler(tmp_path, lambda _: None)
    facts = {
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": [stage]},
    }
    before = deepcopy(facts)
    result = decide_from_saved_facts(
        handler, facts, "解释当前情况", lambda: pytest.fail("Dialogue executed science")
    )

    assert result["status"] == status
    if status == "answered":
        assert result["answer"] == outcome["answer"]
    assert facts == before


def test_approved_recovery_enters_planner_in_the_same_turn(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import decide, scientific_workflow
    from phase_agent.graphs.dialogue.state import TurnRuntime
    from tests.test_unified_react_dialogue import make_handler

    handler = make_handler(tmp_path, lambda _: pytest.fail("Repeated dialogue classification"))
    monkeypatch.setattr("phase_agent.graphs.dialogue.support.fresh_turn_context", lambda *_: {})
    calls = []
    expected = {
        "status": "awaiting_approval",
        "agent_proposal": {"recommended_action": "update_mlip"},
    }

    def plan(invocation, feedback, message):
        calls.append((invocation, feedback, message))
        return expected

    handler._run = plan
    runtime = SimpleNamespace(context=TurnRuntime(handler, [], "same-turn"))
    state = {
        "message": "这些回传现在可以处理，并建议下一步",
        "facts": {"post_recovery_next_step_pending": True},
        "recovery_just_approved": True,
    }

    update = decide(state, runtime)
    assert update["operation"] == "scientific_workflow"
    assert calls == []
    assert scientific_workflow({**state, **update}, runtime)["result"] is expected
    assert len(calls) == 1 and calls[0][1] is None
