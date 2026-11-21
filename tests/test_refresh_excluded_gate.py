import copy

import pytest

from phase_agent.agents.reliability import extrapolation_round
from phase_agent.tools.workflows.model_refresh_gate import advance_refresh


def sample_state():
    accepted = {
        "task_id": "ok",
        "stage": "relax_and_feature",
        "status": "completed",
        "model_version": "m",
        "converged": True,
        "checks_passed": True,
        "model_refresh_id": "plan",
        "input_structure_sha256": "good",
        "outputs": {"optimizer_converged": True},
    }
    failed = {
        "task_id": "failed",
        "stage": "relax_and_feature",
        "status": "completed",
        "model_version": "m",
        "converged": True,
        "checks_passed": False,
        "model_refresh_id": "plan",
        "input_structure_sha256": "bad",
        "result_disposition": "excluded_extrapolation_failure",
        "outputs": {
            "optimizer_converged": True,
            "extrapolation_monitor": {"extrapolation_failure": True},
        },
    }
    return {
        "active_model_version": "m",
        "model_refresh": {
            "status": "waiting_results",
            "new_model_version": "m",
            "wave": 0,
            "task_ids": ["ok", "failed"],
            "plan": {"checksum": "plan", "total_unique": 2, "policy": {}},
        },
        "tasks": [accepted, failed],
        "phase_records": [{"record_id": "r", "source_task_id": "ok", "source_version": "m"}],
        "phase_diagrams": {
            "mlip": {"status": "completed", "model_version": "m", "entries": [{"record_id": "r"}]}
        },
    }


@pytest.mark.parametrize(
    "fraction,expected", [(0.075, "completed"), (0.1, "waiting_results"), (None, "waiting_results")]
)
def test_excluded_results_need_complete_below_threshold_evidence(monkeypatch, fraction, expected):
    status = "complete" if fraction is not None else "unknown"
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda _: {
            "rounds": [
                {
                    "scope_id": "epoch_refresh",
                    "status": status,
                    "failure_fraction": fraction,
                    "direct_sampling_required": fraction is not None and fraction >= 0.1,
                }
            ]
        },
    )
    state, reason = advance_refresh(sample_state())
    assert state["model_refresh"]["status"] == expected
    assert bool(reason) == (expected != "completed")
    if expected == "completed":
        assert state["model_refresh"]["extrapolation_excluded_task_ids"] == ["failed"]
        assert state["model_refresh"]["partial"] is True


def test_wrong_model_exclusion_stays_unresolved(monkeypatch):
    monkeypatch.setattr(
        extrapolation_round,
        "assess_current_rounds",
        lambda _: {
            "rounds": [
                {"scope_id": "epoch_refresh", "status": "complete", "failure_fraction": 0.01}
            ]
        },
    )
    original = sample_state()
    original["tasks"][1]["model_version"] = "other"
    state, reason = advance_refresh(copy.deepcopy(original))
    assert state["model_refresh"]["status"] == "waiting_results"
    assert reason
