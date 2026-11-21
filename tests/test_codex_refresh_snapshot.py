import json
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

from phase_agent.runtime.codex.context import build_codex_context
from phase_agent.runtime.local_agent_control import LocalAgentControl


def make_control(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    return LocalAgentControl(SimpleNamespace(state_path=state, lock=RLock())), None


def test_codex_context_includes_current_refresh_reliability_and_hull(tmp_path):
    control, _ = make_control(tmp_path)
    state = {
        "active_model_version": "new",
        "model_refresh": {"new_model_version": "new", "task_ids": ["t"]},
        "phase_diagrams": {
            "mlip": {"version": "current", "model_version": "new", "status": "completed"}
        },
        "current_phase_diagram": {
            "mlip": {"version": "old", "model_version": "old", "status": "completed"}
        },
    }
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    control.chat_handler.workflow_kwargs = {
        "config_session": {"config": {"agent": {"allowed_tools": ["prepare_local_batch_files"]}}}
    }
    rounds = {
        "model_version": "new",
        "rounds": [
            {
                "scope_id": "epoch_refresh",
                "status": "complete",
                "failure_fraction": 0.118,
                "direct_sampling_required": True,
            }
        ],
    }
    with patch(
        "phase_agent.agents.reliability.extrapolation_round.assess_current_rounds",
        return_value=rounds,
    ):
        context = build_codex_context(control.chat_handler)
    evidence = context["decision_context"]
    assert evidence["reliability_extrapolation_rounds"] == rounds
    refs = evidence["evidence_reference_index"]["references"]
    assert "phase_diagram:mlip:current" in refs
    assert "phase_diagram:mlip:old" not in refs


def test_codex_context_does_not_relabel_old_model_diagram(tmp_path):
    control, _ = make_control(tmp_path)
    state = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {"version": "old", "model_version": "old", "status": "completed"}
        },
    }
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    control.chat_handler.workflow_kwargs = {
        "config_session": {"config": {"agent": {"allowed_tools": []}}}
    }
    context = build_codex_context(control.chat_handler)
    refs = context["decision_context"]["evidence_reference_index"]["references"]
    assert "phase_diagram:mlip:old" not in refs


def test_codex_stage_snapshot_uses_current_model_and_registered_generation(tmp_path):
    from tests.test_new_model_generation_gate import generated

    control, _ = make_control(tmp_path)
    state = generated()
    state["confirmed_config_version"] = "config-current"
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    control.chat_handler.workflow_kwargs = {
        "config_session": {"config": {"agent": {"allowed_tools": []}}}
    }
    context = build_codex_context(control.chat_handler)
    stage = context["decision_context"]["stage_snapshot"]
    assert stage["active_model_version"] == "new"
    assert stage["config_version"] == "config-current"
    assert stage["branch_milestones"]["registration_completed"] is True
    assert stage["branch_milestones"]["registered_branch_count"] == 1
    assert stage["branch_milestones"]["first_search_completed"] is False
