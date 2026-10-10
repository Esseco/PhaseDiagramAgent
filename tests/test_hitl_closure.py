from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from phase_agent.runtime.review_requests import additional_reviews
from phase_agent.tools.state.approved_direction import action_direction, direction_hash
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from tests.test_single_training_approval import setup_direction


@pytest.mark.parametrize("unified", [True, False])
def test_modify_saved_training_plan_then_approve_exact_revision(tmp_path, monkeypatch, unified):
    from tests.test_execution_policy import ExecutionPolicyTest
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    import phase_agent.tools.local.recover_remote_training as recovery

    setup = ExecutionPolicyTest()
    setup.setUp()
    registry = create_tool_registry({"adjust_strategy": setup._handler})
    state = setup_direction()
    old_hash = direction_hash(action_direction(state))
    context = {"state_path": str(tmp_path / "state.json"), "unified_dialogue": unified}

    def forbidden(*args, **kwargs):
        raise AssertionError("approval must use saved plan")

    pending = run_tool_step(
        state,
        setup.session,
        registry=registry,
        execution_mode="interactive",
        invocation_id="edit",
        context=context,
        agent_client=forbidden,
    )
    old_proposal = deepcopy(pending["agent_proposal"])
    requests = []

    def agent(payload):
        requests.append(payload)
        plan = deepcopy(action_direction(state)["plan"])
        plan["expected_purpose"] = "修改后覆盖"
        plan["reason"] = "按反馈调整"
        plan["parameters"] = {"coverage_note": "more"}
        return {
            "reason": "修改后收益更高",
            "parameters": {
                "choice": "supplement_dft",
                "alternatives": "暂缓切换",
                "followup_action": plan,
            },
        }

    revised = run_tool_step(
        pending["state"],
        setup.session,
        registry=registry,
        execution_mode="interactive",
        invocation_id="edit",
        context={**context, "user_message": "增加覆盖"},
        human_feedback={"decision": "comment", "comment": "增加覆盖"},
        agent_client=agent,
    )
    assert revised["status"] == "awaiting_approval", revised
    assert requests[0]["decision_context"]["user_feedback"] == "增加覆盖"
    assert direction_hash(action_direction(revised["state"])) != old_hash
    assert revised["agent_proposal"]["raw_action"]["parameters"] == {"coverage_note": "more"}
    assert revised["state"]["pending_execution_policies"]["edit"]["revision"] == 1
    assert setup.calls == []
    from phase_agent.tools.workflows.react_proposal import stale_proposal_error

    assert stale_proposal_error(old_proposal, revised["state"], context, {})
    monkeypatch.setattr(
        recovery, "inspect_training_results", lambda job: {"fingerprint": "evidence"}
    )
    approved = run_tool_step(
        revised["state"],
        setup.session,
        registry=registry,
        execution_mode="interactive",
        invocation_id="edit",
        context=context,
        human_feedback="approve",
        agent_client=forbidden,
    )
    assert approved["status"] == "completed", approved
    assert len(setup.calls) == 1


def activation_state():
    return {
        "active_model_version": "old",
        "candidate_models": {
            "new": {
                "status": "validated_candidate",
                "old_model_version": "old",
                "agent_review": {"choice": "activate"},
            }
        },
        "remote_finetune_jobs": {
            "job": {
                "training_handoff": {
                    "stage": "awaiting_activation_approval",
                    "candidate_model_version": "new",
                    "direction_proposal": {
                        "direction": "activate",
                        "reason": "收益更高",
                        "training_fingerprint": "f",
                    },
                }
            }
        },
    }


def test_queue_lists_activation_config_and_rejects_changed_identity(tmp_path):
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.runtime.chat_review import review_pending
    from phase_agent.tools.step_runner.file_protocol import write_json
    import threading

    state = activation_state()
    handler = SimpleNamespace(
        state_path=tmp_path / "state.json",
        lock=threading.RLock(),
        workflow_kwargs={"config_session": {"status": "draft", "draft_revision": 2, "config": {}}},
    )
    write_json(handler.state_path, state)
    result = LocalAgentControl(handler).pending()
    assert {row["review_kind"] for row in result["pending"]} == {
        "model_activation",
        "configuration",
    }
    assert LocalAgentControl(handler).status()["pending_approvals"] == 2
    row = next(row for row in result["pending"] if row["review_kind"] == "model_activation")
    with pytest.raises(ValueError, match="confirm_sensitive"):
        review_pending(
            handler,
            row["plan_id"],
            "approve",
            expected_state_version=row["state_version"],
            expected_proposal_hash=row["proposal_hash"],
        )
    state["remote_finetune_jobs"]["job"]["training_handoff"]["direction_proposal"]["reason"] = (
        "新方案"
    )
    write_json(handler.state_path, state)
    with pytest.raises(ValueError, match="review_changed"):
        review_pending(
            handler,
            row["plan_id"],
            "confirm_sensitive",
            expected_state_version=row["state_version"],
            expected_proposal_hash=row["proposal_hash"],
        )


def test_new_conversation_cannot_blindly_approve_model_switch(tmp_path):
    from phase_agent.graphs.dialogue.support import review_presented_proposal

    handler = SimpleNamespace(
        state_path=tmp_path / "state.json", conversation_id="new", review_pending=Mock()
    )
    reply = review_presented_proposal(handler, activation_state(), "同意")
    assert "请核对" in reply
    handler.review_pending.assert_not_called()
    reply = review_presented_proposal(handler, activation_state(), "同意")
    assert "审批页" in reply
    handler.review_pending.assert_not_called()


def test_service_record_mismatch_falls_back_and_unhealthy_can_be_stopped(tmp_path):
    import psutil
    from phase_agent.runtime.project_service import read_service, write_service

    path = tmp_path / "runtime.json"
    command = ["python", "-m", "phase_agent.runtime.studio_service", "--runtime-config", str(path)]
    process = SimpleNamespace(create_time=lambda: 100, cmdline=lambda: command)
    write_service(
        path, {"pid": 12, "process_started": 99, "runtime_config": str(path), "status": "ready"}
    )
    with (
        patch.object(psutil, "Process", return_value=process),
        patch(
            "phase_agent.runtime.project_service.recover_service_identity", return_value={"pid": 13}
        ) as recover,
    ):
        assert read_service(path)["pid"] == 13
        recover.assert_called_once()
    write_service(
        path, {"pid": 12, "process_started": 100, "runtime_config": str(path), "status": "starting"}
    )
    with patch.object(psutil, "Process", return_value=process):
        assert read_service(path, require_ready=False)["pid"] == 12


def test_studio_child_failure_is_saved_with_exit_code_and_log(tmp_path):
    from phase_agent.runtime.studio_service import run_service
    from phase_agent.tools.step_runner.file_protocol import read_json

    args = SimpleNamespace(
        runtime_config=str(tmp_path / "runtime.json"), port=2024, control_port=8765, parent_pid=None
    )
    child = Mock(pid=123)
    child.poll.return_value = 7
    with (
        patch(
            "phase_agent.runtime.studio_service.preflight",
            return_value=(tmp_path / "runtime.json", tmp_path / "langgraph.exe"),
        ),
        patch("phase_agent.runtime.local_project_launcher.validate_project_paths"),
        patch(
            "phase_agent.runtime.project_service.prepare_studio_config",
            return_value=tmp_path / "langgraph.json",
        ),
        patch("phase_agent.runtime.studio_service.subprocess.Popen", return_value=child) as spawn,
    ):
        with pytest.raises(RuntimeError, match="退出码 7"):
            run_service(args)
    record = read_json(tmp_path / "workflow_state/studio/service.json")
    assert record["status"] == "failed"
    assert record["child_exit_code"] == 7
    assert record["child_log"].endswith("studio_server.log")
    assert spawn.call_args.args[0][1:4] == ["-u", "-m", "langgraph_cli"]
    assert spawn.call_args.kwargs["stdout"] is not None


def test_stop_requires_children_to_exit():
    from phase_agent.runtime.local_project_launcher import _stop_owned_process

    parent = Mock()
    parent.pid = 12
    parent.poll.side_effect = [None, 0]
    child = Mock()
    with (
        patch("phase_agent.runtime.studio_run_lifecycle.interrupt_owned_studio"),
        patch("psutil.Process") as factory,
        patch("psutil.wait_procs", side_effect=[([], [child]), ([], [child])]),
    ):
        factory.return_value.children.return_value = [child]
        assert not _stop_owned_process(parent)
    child.kill.assert_called_once()


def test_lifecycle_approval_does_not_reinterpret_control_message():
    from phase_agent.tools.workflows.lifecycle_recovery import _review_training_direction

    state = setup_direction()
    handoff = deepcopy(state["remote_finetune_jobs"]["j"]["training_handoff"])
    frame = {
        "pre_reconciled": {"state": state},
        "training_handoffs": [handoff],
        "human_feedback": {"decision": "approve"},
        "runtime_adapters": {"user_message": "local approval page"},
        "agent_client": Mock(side_effect=AssertionError("approval must not re-review")),
    }
    result = _review_training_direction(frame)
    frame["agent_client"].assert_not_called()
    assert action_direction(result["loaded_state"]) == handoff["direction_proposal"]


def test_invalid_direction_revision_preserves_original_plan():
    from phase_agent.tools.workflows.training_plan_revision import revise_training_plan

    state = setup_direction()
    from phase_agent.tools.local.training_followup_plan import saved_followup

    direction = action_direction(state)
    action = saved_followup(direction)
    action["_approved_direction_hash"] = direction_hash(direction)
    stored = {"agent_proposal": {"raw_action": action}}
    before = deepcopy(state)
    updated, response = revise_training_plan(state, stored, "修改", lambda payload: {})
    assert response["status"] == "rejected"
    assert updated == state == before


def test_modify_direction_to_activation_removes_old_action_without_execution():
    from phase_agent.tools.workflows.training_plan_revision import revise_training_plan

    direction_state = setup_direction()
    direction = action_direction(direction_state)
    action = deepcopy(direction["plan"])
    action["_approved_direction_hash"] = direction_hash(direction)
    stored = {"agent_proposal": {"raw_action": action}}
    direction_state["pending_execution_policies"] = {"old": stored}
    updated, response = revise_training_plan(
        direction_state,
        stored,
        "改为切换",
        lambda payload: {
            "reason": "搜索收益更高",
            "parameters": {"choice": "activate", "alternatives": "暂缓补DFT"},
        },
    )
    assert response["status"] == "training_handoff"
    assert not updated["pending_execution_policies"]
    assert updated["active_model_version"] == "old"


def test_http_and_studio_use_identical_review_identity(tmp_path):
    from phase_agent.runtime.review_requests import pending_reviews
    from phase_agent.runtime.scientific_progress import project_progress

    state = activation_state()
    assert project_progress(state, tmp_path / "state.json")["review_requests"] == pending_reviews(
        state
    )


def test_healthy_endpoint_from_another_project_is_not_reused(tmp_path):
    import psutil
    from phase_agent.runtime.project_service import read_service, write_service

    config = tmp_path / "runtime.json"
    value = {
        "pid": 12,
        "process_started": 100,
        "runtime_config": str(config),
        "status": "ready",
        "studio_port": 2025,
        "control_port": 8766,
    }
    write_service(config, value)
    process = SimpleNamespace(
        pid=12,
        create_time=lambda: 100,
        children=lambda recursive: [],
        cmdline=lambda: [
            "python",
            "-m",
            "phase_agent.runtime.studio_service",
            "--runtime-config",
            str(config),
            "--port",
            "2025",
            "--control-port",
            "8766",
        ],
    )
    with (
        patch.object(psutil, "Process", return_value=process),
        patch.object(
            psutil,
            "net_connections",
            return_value=[
                SimpleNamespace(
                    pid=99, status="LISTEN", laddr=SimpleNamespace(ip="127.0.0.1", port=2025)
                )
            ],
        ),
        patch("phase_agent.runtime.project_service.service_healthy", return_value=True),
        patch("phase_agent.runtime.project_service.recover_service_identity", return_value=None),
    ):
        assert read_service(config) is None
        assert read_service(config, require_ready=False)["pid"] == 12
