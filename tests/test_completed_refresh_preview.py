from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.tools.workflows.preview_completed_refresh import preview_completed_refresh
from phase_agent.tools.workflows import model_refresh_state


@pytest.fixture
def state():
    baseline = {"status": "completed", "version": "baseline", "model_version": "old"}
    return {"active_model_version": "new", "phase_diagrams": {"mlip": {"version": "hull"}},
            "model_refresh": {"status": "completed", "new_model_version": "new",
                              "plan": {"checksum": "plan"}, "task_ids": ["t"]},
            "tasks": [{"task_id": "t", "status": "completed"}],
            "extrapolation_recovery_cycle": {"policy_version": "epoch_refresh_v1",
                 "status": "completed", "replacement_model_version": "new",
                 "replacement_hull_version": "hull", "baseline_model_version": "old",
                 "baseline_hull_version": "baseline", "baseline_diagram": baseline}}


def invoke(state):
    return preview_completed_refresh(state, SimpleNamespace(), {}, expected_model="new",
                                     expected_plan="plan", expected_hull="hull")


def test_preview_uses_original_baseline_and_does_not_change_state(state, monkeypatch):
    before = deepcopy(state)
    def fake(view, manager, config):
        assert view["model_refresh"]["old_diagram"] == before["extrapolation_recovery_cycle"]["baseline_diagram"]
        assert view["extrapolation_recovery_cycle"]["status"] == "awaiting_new_model_refresh"
        return {"checksum": "new-plan", "candidates": []}
    monkeypatch.setattr(model_refresh_state, "refresh_preview", fake)
    assert invoke(state)["plan"]["checksum"] == "new-plan"
    assert state == before


@pytest.mark.parametrize("change", [
    lambda s: s.update(active_model_version="wrong"),
    lambda s: s["model_refresh"]["plan"].update(checksum="wrong"),
    lambda s: s["phase_diagrams"]["mlip"].update(version="wrong"),
    lambda s: s.update(pending_execution_policies={"x": {}}),
    lambda s: s["tasks"][0].update(status="running"),
    lambda s: s["extrapolation_recovery_cycle"].update(baseline_hull_version="wrong"),
])
def test_rejects_stale_or_unsettled_refresh(state, monkeypatch, change):
    change(state)
    monkeypatch.setattr(model_refresh_state, "refresh_preview", lambda *args: pytest.fail("unexpected preview"))
    with pytest.raises(ValueError):
        invoke(state)


def test_mcp_dispatches_explicit_preview_bindings():
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    class FakeCodex:
        def preview_completed_refresh(self, **kwargs):
            return {"status": "preview_only", "bindings": kwargs}

    server = CodexMcpServer(FakeCodex())
    result = server._call_tool("phase_preview_completed_refresh", {
        "expected_model": "new", "expected_plan": "plan", "expected_hull": "hull"
    })
    assert result["status"] == "preview_only"
    assert result["bindings"]["expected_hull"] == "hull"


def test_normal_completed_refresh_preserves_baseline_and_fences_old_task_ids(state, monkeypatch):
    from phase_agent.tools.workflows.reopen_completed_refresh import reopen_state

    cycle = state.pop("extrapolation_recovery_cycle")
    state["model_refresh"].update(old_model_version="old", old_diagram=cycle["baseline_diagram"])
    before = deepcopy(state)

    def preview(view, manager, config):
        assert view["model_refresh"]["reopened_from_plan"] == "plan"
        assert view["completed_refresh_replans"][-1]["prior_refresh"] == before["model_refresh"]
        return {
            "checksum": "new-plan", "new_model_version": "new", "reopened_from_plan": "plan",
            "hd_refresh_reference": {"hull_version": "baseline"}, "candidates": [],
        }

    monkeypatch.setattr(model_refresh_state, "refresh_preview", preview)
    reopened = reopen_state(
        state, invoke(state), expected_model="new", expected_plan="plan",
        expected_hull="hull", expected_preview="new-plan",
    )
    assert state == before
    assert reopened["tasks"] == before["tasks"]
    assert "task_ids" not in reopened["model_refresh"]
    assert "extrapolation_recovery_cycle" not in reopened
    assert reopened["phase_diagrams"]["mlip"]["status"] == "not_ready"
