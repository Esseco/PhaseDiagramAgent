from types import SimpleNamespace
from threading import RLock
from unittest.mock import Mock, patch

from phase_agent.runtime.codex.offline import CodexOffline


def test_observe_reuses_runtime_feedback_without_decision(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text('{"tasks":[]}', encoding="utf-8")
    manager, collector = object(), object()
    handler = SimpleNamespace(
        state_path=state_path,
        lock=RLock(),
        workflow_kwargs={"manager": manager, "result_collector": collector},
    )
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=Mock()):
        codex = CodexOffline("runtime.json", runtime_factory=lambda _: handler)
    with patch(
        "phase_agent.tools.workflows.recover_saved_results.recover_saved_results",
        return_value={"dft_dataset_records": [{"task_id": "T"}]},
    ) as recover:
        result = codex.recover()
    assert result["dataset_count"] == 1
    assert result["submitted"] is False
    assert recover.call_args.kwargs["manager"] is manager
    assert recover.call_args.kwargs["result_collector"] is collector
    codex.control.propose_codex.assert_not_called()
    codex.control.decide.assert_not_called()


def test_observe_registers_training_returns_without_activation(tmp_path):
    import json

    state_path = tmp_path / "state.json"
    state = {"remote_finetune_jobs": {"J": {"status": "inputs_prepared", "activated": False}}}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    handler = SimpleNamespace(state_path=state_path, lock=RLock(), workflow_kwargs={})
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=Mock()):
        codex = CodexOffline("runtime.json", runtime_factory=lambda _: handler)
    received = {"remote_finetune_jobs": {"J": {"status": "results_received", "activated": False}}}
    reports = [{"status": "validation_required", "issues": []}]
    with (
        patch(
            "phase_agent.tools.workflows.recover_saved_results.recover_saved_results",
            return_value=state,
        ),
        patch(
            "phase_agent.tools.local.recover_remote_training.recover_remote_training",
            return_value=(received, reports),
        ) as inspect,
        patch("phase_agent.tools.workflows.lifecycle_support._save_runner_state") as save,
    ):
        result = codex.recover()
    inspect.assert_called_once_with(state)
    save.assert_called_once_with(received, state_path)
    assert result["training_returns"] == reports
    assert received["remote_finetune_jobs"]["J"]["activated"] is False
    assert result["submitted"] is False
    codex.control.propose_codex.assert_not_called()
    codex.control.decide.assert_not_called()


def test_observe_resumes_original_training_handoff(tmp_path):
    import json

    state = {"confirmed_config": {"system": {}}, "confirmed_config_version": "C"}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    handler = SimpleNamespace(
        state_path=state_path, lock=RLock(), workflow_kwargs={"run_config": {}}
    )
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=Mock()):
        codex = CodexOffline("runtime.json", runtime_factory=lambda _: handler)
    received = {**state, "remote_finetune_jobs": {"J": {"status": "results_received"}}}
    handoffs = [{"stage": "agent_review_required", "submitted": False}]
    handed = {**received, "candidate_models": {"candidate": {"activated": False}}}
    with (
        patch(
            "phase_agent.tools.workflows.recover_saved_results.recover_saved_results",
            return_value=state,
        ),
        patch(
            "phase_agent.tools.local.recover_remote_training.recover_remote_training",
            return_value=(received, [{"status": "validation_required"}]),
        ),
        patch(
            "phase_agent.configuration.runtime.build_effective_run_config.build_effective_run_config",
            return_value={"effective": True},
        ),
        patch(
            "phase_agent.tools.local.training_handoff.advance_training_handoffs",
            return_value=(handed, handoffs),
        ) as advance,
        patch("phase_agent.tools.workflows.lifecycle_support._save_runner_state") as save,
    ):
        result = codex.recover()
    advance.assert_called_once_with(received, {"effective": True}, state_path=state_path)
    save.assert_called_once_with(handed, state_path)
    assert result["training_handoffs"] == handoffs
    assert result["submitted"] is False
    codex.control.propose_codex.assert_not_called()
    codex.control.decide.assert_not_called()


def test_observe_settles_refresh_through_original_gate(tmp_path):
    import json

    state = {
        "active_model_version": "M",
        "model_refresh": {"status": "waiting_results", "new_model_version": "M"},
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    handler = SimpleNamespace(state_path=state_path, lock=RLock(), workflow_kwargs={})
    with patch("phase_agent.runtime.codex.offline.LocalAgentControl", return_value=Mock()):
        codex = CodexOffline("runtime.json", runtime_factory=lambda _: handler)
    complete = {**state, "model_refresh": {**state["model_refresh"], "status": "completed"}}
    with (
        patch(
            "phase_agent.tools.workflows.recover_saved_results.recover_saved_results",
            return_value=state,
        ),
        patch(
            "phase_agent.tools.local.recover_remote_training.recover_remote_training",
            return_value=(state, []),
        ),
        patch(
            "phase_agent.tools.workflows.model_refresh_gate.advance_refresh",
            return_value=(complete, None),
        ) as advance,
        patch("phase_agent.tools.workflows.lifecycle_support._save_runner_state") as save,
    ):
        result = codex.recover()
    advance.assert_called_once_with(state)
    save.assert_called_once_with(complete, state_path)
    assert result["model_refresh_status"] == "completed"
    assert result["model_refresh_waiting_reason"] is None
    codex.control.propose_codex.assert_not_called()
    codex.control.decide.assert_not_called()
