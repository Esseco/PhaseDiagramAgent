from copy import deepcopy

import pytest

from phase_agent.tools.local.training_agent_review import REVIEW_POLICY, review_training_choice
from phase_agent.tools.state.approved_direction import action_direction, direction_hash
from phase_agent.tools.workflows.run_tool_step import run_tool_step


def setup_direction():
    plan = {
        "tool": "adjust_strategy",
        "parameters": {},
        "target_ids": [],
        "budget": 0.0,
        "reason": "补采样前调整策略",
        "expected_purpose": "增加覆盖",
    }
    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "training_fingerprint": "evidence",
        "evidence_type": "grouped_cross_validation_review",
    }
    state = {
        "active_model_version": "old",
        "candidate_models": {"new": {"validation": {}}},
        "remote_finetune_jobs": {
            "j": {
                "original_model_version": "old",
                "status": "results_received",
                "training_handoff": deepcopy(handoff),
            }
        },
    }

    def agent(payload):
        return {
            "reason": "补覆盖收益更高",
            "parameters": {
                "choice": "supplement_dft",
                "alternatives": "暂缓切换",
                "followup_action": plan,
            },
        }

    state, waits = review_training_choice(state, [handoff], agent)
    assert waits[0]["stage"] == "execution_plan_ready"
    assert "回复‘同意’" not in waits[0]["reason"]
    return state


@pytest.mark.parametrize("unified", [True, False])
def test_single_approval_executes_saved_action_and_replay_is_safe(tmp_path, monkeypatch, unified):
    from tests.test_execution_policy import ExecutionPolicyTest
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.graphs.direction_review_graph import direction_checkpoint
    import phase_agent.tools.local.recover_remote_training as recovery

    setup = ExecutionPolicyTest()
    setup.setUp()
    registry = create_tool_registry({"adjust_strategy": setup._handler})
    state = setup_direction()
    proposal = action_direction(state)
    assert proposal["review_policy"] == REVIEW_POLICY
    context = {"state_path": str(tmp_path / "state.json"), "unified_dialogue": unified}

    def forbidden(*args, **kwargs):
        raise AssertionError("saved action must not call Agent again")

    monkeypatch.setattr(
        recovery, "inspect_training_results", lambda job: {"fingerprint": "evidence"}
    )
    pending = run_tool_step(
        state,
        setup.session,
        registry=registry,
        execution_mode="interactive",
        invocation_id="one-plan",
        context=context,
        agent_client=forbidden,
    )
    assert pending["status"] == "awaiting_approval", pending
    assert setup.calls == []
    assert pending["agent_proposal"]["raw_action"]["parameters"] == proposal["plan"]["parameters"]
    completed = run_tool_step(
        pending["state"],
        setup.session,
        registry=registry,
        execution_mode="interactive",
        invocation_id="one-plan",
        context=context,
        human_feedback="approve",
        agent_client=forbidden,
    )
    assert completed["status"] == "completed", completed
    assert len(setup.calls) == 1
    assert direction_checkpoint(context["state_path"], proposal)["status"] == "approved"
    assert (
        completed["state"]["candidate_models"]["new"]["agent_review"]["followup_result"]["status"]
        == "completed"
    )
    replay = run_tool_step(
        completed["state"],
        setup.session,
        registry=registry,
        execution_mode="interactive",
        invocation_id="one-plan",
        context=context,
        human_feedback="approve",
        agent_client=forbidden,
    )
    assert replay["idempotent_replay"]
    assert len(setup.calls) == 1
    handoff = deepcopy(completed["state"]["remote_finetune_jobs"]["j"]["training_handoff"])
    handoff["stage"] = "awaiting_activation_approval"  # Simulate the next CV result recovery.
    _, waits = review_training_choice(completed["state"], [handoff], forbidden)
    assert waits == []


def test_changed_training_evidence_cannot_approve(tmp_path, monkeypatch):
    from phase_agent.tools.state.approved_direction import record_action_direction
    import phase_agent.tools.local.recover_remote_training as recovery

    state = setup_direction()
    proposal = action_direction(state)
    action = {**proposal["plan"], "_approved_direction_hash": direction_hash(proposal)}
    monkeypatch.setattr(
        recovery, "inspect_training_results", lambda job: {"fingerprint": "changed"}
    )
    with pytest.raises(ValueError, match="training_evidence_changed"):
        record_action_direction(state, action, tmp_path / "state.json", "approve")
    assert action_direction(state) == proposal


def test_changed_plan_requires_new_review(tmp_path):
    from phase_agent.tools.state.approved_direction import record_action_direction

    state = setup_direction()
    proposal = action_direction(state)
    action = {
        **proposal["plan"],
        "budget": 100,
        "_approved_direction_hash": direction_hash(proposal),
    }
    with pytest.raises(ValueError, match="saved_direction_plan_changed:budget"):
        record_action_direction(state, action, tmp_path / "state.json", "approve")


def test_rejection_is_persisted_and_does_not_represent_the_plan(tmp_path, monkeypatch):
    from phase_agent.tools.state.approved_direction import record_action_direction
    from phase_agent.graphs.direction_review_graph import direction_checkpoint
    import phase_agent.tools.local.recover_remote_training as recovery

    state = setup_direction()
    proposal = action_direction(state)
    action = {**proposal["plan"], "_approved_direction_hash": direction_hash(proposal)}
    monkeypatch.setattr(
        recovery, "inspect_training_results", lambda job: {"fingerprint": "evidence"}
    )
    record_action_direction(state, action, tmp_path / "state.json", "reject")
    assert direction_checkpoint(tmp_path / "state.json", proposal)["status"] == "rejected"
    handoff = deepcopy(state["remote_finetune_jobs"]["j"]["training_handoff"])
    handoff["stage"] = "awaiting_activation_approval"

    def forbidden(payload):
        raise AssertionError("rejection must not request the same plan again")

    _, waits = review_training_choice(state, [handoff], forbidden)
    assert waits == []
    assert action_direction(state) is None


def test_ready_plan_passes_lifecycle_wait_gate_without_direction_approval():
    from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate

    state = setup_direction()
    handoff = state["remote_finetune_jobs"]["j"]["training_handoff"]
    handoff["cv_review"] = {"status": "completed"}
    frame = {
        "recovery_question": None,
        "execution_mode": "interactive",
        "feedback": {"state": state},
        "state_path": None,
        "effective_config": {},
        "recovered_count": 0,
        "collection_report": None,
        "snapshot": {},
        "pre_reconciled": {},
        "manual_wait": None,
        "rebuilding": False,
        "training_handoffs": [handoff],
    }
    result = _workflow_wait_gate(frame)
    assert result is frame
    assert "status" not in result
    assert "统一展示" in state["training_validation_planning"]["instruction"]
