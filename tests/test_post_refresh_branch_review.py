from copy import deepcopy

import pytest

from tests.test_new_model_generation_gate import switched, generated
from phase_agent.tools.policy.model_generation import (
    new_model_generation_facts,
    new_model_search_errors,
)
from phase_agent.tools.policy.branch_conditions import branch_entry_errors


def test_refresh_requires_actual_generation_before_mc_or_dft():
    state = switched()
    assert new_model_generation_facts(state)["required"]
    assert new_model_search_errors(state, ["old_branch"]) == [
        "new_model_branch_generation_required"
    ]
    assert new_model_search_errors(state, dft=True) == ["new_model_branch_generation_required"]
    assert new_model_search_errors(generated(), ["old_branch"]) == []


@pytest.mark.parametrize(
    "parameters",
    [
        {"module_execution_scope": {"max_registered_branches": 12}},
        {"sampling_mode": "region_adaptive", "regions": [], "defer_reason": "reuse"},
    ],
)
def test_first_generation_cannot_bundle_inputs_or_defer_generation(parameters):
    errors = branch_entry_errors(
        {"tool": "generate_branches", "parameters": parameters}, switched()
    )
    assert "post_refresh_branch_generation_separate_review_required" in errors


def test_saved_mc_plan_is_visible_but_not_invited_for_approval():
    from phase_agent.runtime.plan_queries import pending_plan_reply
    from phase_agent.runtime.status_presentation import format_progress

    state = switched()
    state["pending_execution_policies"] = {
        "old-mc": {
            "agent_proposal": {
                "recommended_action": "allocate_mc_bohb",
                "raw_action": {
                    "tool": "allocate_mc_bohb",
                    "target_ids": ["old_branch"],
                    "parameters": {},
                },
            }
        }
    }
    before = deepcopy(state)
    for reply in (pending_plan_reply(state), format_progress(state)):
        assert "暂不可批准" in reply and "先生成" in reply and "branch" in reply
        assert "回复‘同意’" not in reply and "回复“同意”" not in reply
    assert state == before


def test_refresh_status_names_required_generation_and_review():
    from phase_agent.runtime.chat_state_presentation import brief_chat_state
    from phase_agent.analysis.state.training_context import post_refresh_search_context

    text = brief_chat_state(switched())
    assert "先生成" in text and "branch" in text
    instruction = post_refresh_search_context(switched())["instruction"]
    assert "先生成" in instruction and "单独审批" in instruction
    assert "不强制生成" not in instruction


def test_saved_mc_approval_revalidates_before_any_handler(tmp_path):
    from tests.test_event_loop import _session
    from phase_agent.tools.workflows.run_event_loop import run_action_turn

    state = switched()
    proposal = {
        "recommended_action": "allocate_mc_bohb",
        "raw_action": {
            "tool": "allocate_mc_bohb",
            "target_ids": ["old_branch"],
            "parameters": {},
            "budget": 0,
            "task_key": "old-mc",
            "reason": "historical proposal",
        },
    }
    state["pending_execution_policies"] = {
        "old-mc": {"agent_proposal": proposal, "record_id": "old-mc"}
    }
    calls = []
    result = run_action_turn(
        state,
        _session(),
        registry={"allocate_mc_bohb": {"handler": lambda **kw: calls.append(kw)}},
        agent_client=lambda _: pytest.fail("Saved approval must not call model"),
        context={"unified_dialogue": True},
        invocation_id="old-mc",
        execution_mode="interactive",
        human_feedback={"decision": "approve", "comment": "同意"},
    )
    assert result["status"] == "rejected" and not calls
    assert "new_model_branch_generation_required" in str(result)


def test_unresolved_phase_status_is_a_blocker_not_running_work():
    from phase_agent.runtime.chat_state_presentation import brief_chat_state

    state = switched()
    state["model_refresh"].update(status="waiting_results", task_ids=["t"], wave=0)
    state["tasks"] = [
        {
            "task_id": "t",
            "status": "completed",
            "stage": "relax_and_feature",
            "model_version": "new",
            "phase_identification_status": "unknown",
        }
    ]
    text = brief_chat_state(state)
    assert "相识别受阻" in text
    assert "正在" not in text


@pytest.mark.parametrize("historical_mc", [False, True])
def test_model_context_only_offers_first_generation_or_pause(monkeypatch, historical_mc):
    from phase_agent.tools.workflows import initial_tool_proposal as module
    from tests.test_event_loop import _session

    # Exercise the production selection interface, stopping at its first model call.
    seen = []

    def propose(summary, allowed_tools, agent_client=None, **kwargs):
        seen.append(allowed_tools)
        raise RuntimeError("stop after observing allowed tools")

    state = switched()
    monkeypatch.setattr(
        "phase_agent.science.mc.second_round_state.second_round_completed", lambda *a: historical_mc
    )
    with pytest.raises(RuntimeError, match="stop after"):
        module.select_initial_proposal(
            current=state,
            mode="interactive",
            config=_session()["confirmed_snapshot"]["config"],
            context={"unified_dialogue": True},
            decision_state={},
            registry={
                name: {"handler": lambda **kw: None}
                for name in ["generate_branches", "allocate_mc_bohb", "pause_search"]
            },
            agent_client=lambda _: {},
            invocation_id="x",
            propose=propose,
            revise=lambda *a, **kw: None,
            prepare_debug=lambda *a, **kw: None,
            is_verified_second=lambda *a, **kw: False,
            mc_block=lambda *a, **kw: None,
            model_failed=lambda *a, **kw: False,
            record_id_factory=lambda *a: "x",
        )
    assert seen and "generate_branches" in seen[0] and "allocate_mc_bohb" not in seen[0]


@pytest.mark.parametrize("message", ["同意", "拒绝"])
def test_missing_approval_does_not_collect_results_or_start_a_model(tmp_path, monkeypatch, message):
    from unittest.mock import Mock
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    write_json(path, switched())
    forbidden = Mock(side_effect=AssertionError("No model/science for unavailable approval"))
    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recover_saved_results", forbidden
    )
    handler = RunWorkflowChatHandler({"state_path": str(path), "agent_client": forbidden})
    handler._run = forbidden
    reply = handler([{"role": "user", "content": message}], conversation_id="t")
    assert "没有可批准或拒绝" in reply
    forbidden.assert_not_called()


@pytest.mark.parametrize("valid_evidence", [True, False])
def test_confirmed_outside_boundary_result_is_retained_without_blocking_refresh(valid_evidence):
    from tests.test_refresh_completion_reply import refreshed_state
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    state = refreshed_state()
    state["model_refresh"]["task_ids"].append("outside")
    evidence = {
        "status": "outside_boundary",
        "reason": "phase_outside_search_boundary",
        "observed_phase": "OP3",
        "structure_sha256": "hash",
        "classification_scope": "scope",
    }
    state["tasks"].append(
        {
            "task_id": "outside",
            "model_version": "new",
            "status": "completed",
            "model_refresh_id": "p",
            "input_structure_sha256": "s2",
            "phase_identification_status": "outside_boundary",
            "outputs": {"single_point_completed": True, "phase_identification": evidence},
        }
    )
    state["phase_records"].append(
        {
            "record_id": "outside-record",
            "source_task_id": "outside",
            "source_version": "new",
            "status": "outside_search_boundary",
            "phase_identification_status": "outside_boundary",
            "phase_identification": deepcopy(evidence),
        }
    )
    if not valid_evidence:
        state["phase_records"][-1]["phase_identification"]["structure_sha256"] = "different"
    response = run_model_refresh_gate(state, {}, registry={}, context={})
    if valid_evidence:
        assert response["status"] == "completed"
        assert response["completed_refresh"]["outside_boundary_count"] == 1
        assert len(response["state"]["tasks"]) == 2
        assert response["state"]["phase_diagrams"]["mlip"]["entries"] == [{"record_id": "r1"}]
    else:
        assert response["status"] != "completed"
        assert "未进入新版本相图" in response["reason"]
