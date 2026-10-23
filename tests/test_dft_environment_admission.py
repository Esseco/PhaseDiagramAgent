"""Operational environments stay explicit and are checked before effects or reranking."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from phase_agent.configuration.runtime.build_effective_run_config import build_effective_run_config
from phase_agent.configuration.schema.python_environments import (
    dft_environment_error,
    remote_comparison_model,
)
from phase_agent.tools.workflows.prepare_dft_proposal import prepare_dft_proposal


def selection():
    return {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]},
    }


@pytest.mark.parametrize("saved", [None, {}, {"remote_mlip": None}])
def test_legacy_snapshot_accepts_explicit_runtime_environment_without_rewriting(saved):
    config = {"mlip": {"name": "original-model"}}
    if saved is not None:
        config["python_environments"] = saved
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {"config_version": "v1", "config": config},
    }
    runtime = {"python_environments": {"remote_mlip": "previously-used-hpc-env"}}
    before = deepcopy((session, runtime))
    effective = build_effective_run_config(session, runtime)
    model = remote_comparison_model(
        {"version": "original-model", "environment": "local-only"}, effective
    )
    assert model == {"version": "original-model", "environment": "previously-used-hpc-env"}
    assert (session, runtime) == before
    assert effective["runtime_keys"] == ["python_environments"]


def test_confirmed_remote_environment_wins_and_local_name_is_never_used():
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": "v1",
            "config": {"python_environments": {"remote_mlip": "confirmed-hpc"}},
        },
    }
    effective = build_effective_run_config(
        session, {"python_environments": {"remote_mlip": "other-hpc", "local_mlip": "local-mace"}}
    )
    assert remote_comparison_model({}, effective)["environment"] == "confirmed-hpc"
    with pytest.raises(ValueError, match="remote_mlip"):
        remote_comparison_model(
            {"environment": "mace"}, {"python_environments": {"local_mlip": "mace"}}
        )


def test_project_runtime_carries_saved_environment_to_effective_dft_config(tmp_path, monkeypatch):
    from tests.test_agent_api import OpenWebUIAPITest
    from phase_agent.runtime.agent_api import create_agent_runtime

    runtime = OpenWebUIAPITest()._runtime_files(tmp_path)
    settings = json.loads(runtime.read_text(encoding="utf-8"))
    settings["python_environments"] = {"remote_mlip": "verified-hpc"}
    runtime.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(
        "phase_agent.runtime.runtime_clients.create_optional_runtime_client", lambda *a: None
    )
    handler = create_agent_runtime(runtime)
    kwargs = handler.workflow_kwargs
    effective = build_effective_run_config(kwargs["config_session"], kwargs["run_config"])
    assert remote_comparison_model({}, effective)["environment"] == "verified-hpc"
    assert "python_environments" not in kwargs["config_session"]["confirmed_snapshot"]["config"]


def test_missing_environment_rejects_before_preview_or_model_revision(monkeypatch):
    preview = Mock(side_effect=AssertionError("no costly candidate validation"))
    revise = Mock(side_effect=AssertionError("configuration cannot be fixed by reranking"))
    monkeypatch.setattr(
        "phase_agent.tools.workflows.prepare_dft_proposal.attach_dft_preview", preview
    )
    action, state = selection(), {"qbc_candidates": [{"candidate_id": "s"}]}
    before = deepcopy((action, state))
    result = prepare_dft_proposal(
        action,
        state,
        {},
        {"upload_batches_directory": "uploads"},
        agent_client=Mock(),
        revise=revise,
    )
    assert result.status == "rejected" and "remote_mlip" in result.error
    assert result.action == action and (action, state) == before
    preview.assert_not_called()
    revise.assert_not_called()


def test_analytical_selection_and_other_tools_do_not_require_remote_environment():
    assert dft_environment_error(selection(), {}) is None
    assert (
        dft_environment_error({"tool": "check_convergence"}, {"upload_batches_directory": "u"})
        is None
    )
    assert (
        dft_environment_error(
            selection(),
            {"upload_batches_directory": "u", "python_environments": {"remote_mlip": "current"}},
        )
        is None
    )


def test_approved_action_is_rejected_before_execution_claim(monkeypatch):
    from phase_agent.tools.workflows.run_tool_step import _validation_graph_node

    monkeypatch.setattr(
        "phase_agent.tools.workflows.run_tool_step.validate_tool_action",
        lambda *a: {"valid": True, "errors": [], "config_version": "v1"},
    )
    frame = {
        "current": {},
        "policy": {"final_action": selection(), "execute": True},
        "session": {},
        "registry": {},
        "mode": "interactive",
        "config": {},
        "decision_state": {},
        "explicit_mode": True,
        "execute": True,
        "context": {"effective_config": {"upload_batches_directory": "u"}},
    }
    result = _validation_graph_node(frame)
    assert result["status"] == "rejected" and not result["dispatch_ready"]
    assert "remote_mlip" in result["validation"]["errors"][0]
    assert result["execution"] is None


def test_direct_dft_preparation_leaves_no_directory_or_state_changes(tmp_path, monkeypatch):
    from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches

    reuse = Mock(side_effect=AssertionError("no result ledger work before environment admission"))
    monkeypatch.setattr("phase_agent.persistence.shared_dft.reusable_results", reuse)
    state = {
        "tasks": [
            {"task_id": "d1", "task_key": "k", "stage": "dft_single_point", "status": "pending"}
        ]
    }
    before = deepcopy(state)
    root = tmp_path / "uploads"
    with pytest.raises(ValueError, match="remote_mlip"):
        prepare_dft_upload_batches(
            action={"target_ids": ["d1"]},
            context={
                "event_state": state,
                "effective_config": {"upload_batches_directory": str(root)},
                "manager": None,
            },
            _owned=True,
        )
    assert state == before and not root.exists()
    reuse.assert_not_called()


def test_doctor_shows_actual_environment_source_read_only(tmp_path):
    from tests.test_runtime_doctor import project, files, write
    from phase_agent.runtime.project_diagnostics import inspect_runtime

    runtime = project(tmp_path)
    settings = json.loads(runtime.read_text(encoding="utf-8-sig"))
    settings["python_environments"] = {"remote_mlip": "historical-hpc-env"}
    write(runtime, settings)
    before = files(tmp_path)
    report = inspect_runtime(runtime)
    env = report["sections"]["python_environments"]["data"]
    assert env["values"]["remote_mlip"] == "historical-hpc-env"
    assert env["sources"]["remote_mlip"] == "agent_runtime.json"
    assert env["sources"]["local_python"] == "draft_config"
    assert env["missing"] == ["remote_python"] and not env["remote_checked"]
    assert files(tmp_path) == before


@pytest.mark.parametrize("value", ["mace", ["mace"], 42])
def test_invalid_environment_container_is_actionable(value):
    error = dft_environment_error(
        selection(), {"upload_batches_directory": "u", "python_environments": value}
    )
    assert "python_environments" in error
