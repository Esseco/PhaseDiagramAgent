from copy import deepcopy
import pytest
from phase_agent.runtime.phase_diagram_chat import _active_model_plot_state


def test_active_snapshot_replaces_mixed_without_mutating_state(tmp_path):
    archive = tmp_path / "model.csv"
    archive.write_text("x")
    mixed = {
        "model_version": "new",
        "version": "mixed",
        "entries": [{"source_version": "old"}, {"source_version": "new"}],
    }
    active = {
        "model_version": "new",
        "version": "pure",
        "archive_csv_path": str(archive),
        "csv_path": "global.csv",
        "entries": [{"source_version": "new"}],
    }
    state = {
        "active_model_version": "new",
        "phase_diagrams": {"mlip": mixed},
        "phase_diagrams_by_model": {"new": active},
    }
    before = deepcopy(state)
    result = _active_model_plot_state(state)
    assert result["phase_diagrams"]["mlip"]["version"] == "pure"
    assert result["phase_diagrams"]["mlip"]["csv_path"] == str(archive)
    assert state == before


def test_mixed_snapshot_without_independent_model_fails():
    state = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {
                "model_version": "new",
                "entries": [{"source_version": "old"}, {"source_version": "new"}],
            }
        },
    }
    with pytest.raises(ValueError, match="独立相图"):
        _active_model_plot_state(state)


def test_legacy_unmixed_diagram_remains_valid():
    state = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {"model_version": "new", "entries": [{"source_version": "new"}]}
        },
    }
    assert _active_model_plot_state(state) is state


def test_decision_context_uses_active_independent_diagram():
    from phase_agent.evidence.build_decision_context import build_decision_context

    s = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {
                "version": "mixed",
                "model_version": "new",
                "entries": [{"source_version": "old"}, {"source_version": "new"}],
            }
        },
        "phase_diagrams_by_model": {
            "new": {"version": "pure", "status": "completed", "model_version": "new", "entries": []}
        },
    }
    context = build_decision_context(s)
    assert context["current_phase_diagram"]["mlip"]["version"] == "pure"
    assert "phase_diagram:mlip:pure" in context["evidence_catalog"]["references"]


def test_new_model_pending_refresh_does_not_relabel_old_hull():
    from phase_agent.evidence.model_diagram_scope import active_model_diagram

    state = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {
                "model_version": "old",
                "version": "hull",
                "entries": [{"source_version": "old"}],
            }
        },
        "model_refresh": {"new_model_version": "new", "status": "approval_required"},
    }
    result = active_model_diagram(state)
    assert result == {
        "status": "pending_refresh",
        "model_version": "new",
        "reason": "current_model_refresh_not_completed",
    }
    assert state["phase_diagrams"]["mlip"]["model_version"] == "old"


def test_pending_refresh_rejects_relabelled_old_entries():
    from phase_agent.evidence.model_diagram_scope import active_model_diagram

    state = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {"model_version": "new", "entries": [{"source_version": "old"}]}
        },
        "model_refresh": {"new_model_version": "new", "status": "approval_required"},
    }
    assert active_model_diagram(state)["status"] == "pending_refresh"


def test_decision_uses_latest_per_structure_mixed_hull_without_changing_independent_scope():
    from phase_agent.evidence.build_decision_context import build_decision_context
    from phase_agent.evidence.build_state_snapshot import build_state_snapshot
    from phase_agent.evidence.model_diagram_scope import active_model_diagram

    latest = {
        "version": "latest",
        "status": "completed",
        "model_version": "new",
        "energy_policy": "latest_valid_per_structure_v1",
        "mixed_model_versions": True,
        "independent_model_comparability": "not_established",
        "entries": [],
    }
    independent = {"version": "pure", "status": "completed", "model_version": "new", "entries": []}
    state = {
        "active_model_version": "new",
        "phase_diagrams": {"mlip": latest},
        "phase_diagrams_by_model": {"new": independent},
    }
    before = deepcopy(state)
    context = build_decision_context(state)
    assert context["current_phase_diagram"]["mlip"]["version"] == "latest"
    assert context["current_phase_diagram"]["mlip"]["mixed_model_versions"] is True
    assert (
        context["current_phase_diagram"]["mlip"]["independent_model_comparability"]
        == "not_established"
    )
    assert "phase_diagram:mlip:latest" in context["evidence_catalog"]["references"]
    assert (
        build_state_snapshot(state, snapshot_index=1)["current_convex_hull"]["mlip"]["version"]
        == "latest"
    )
    assert active_model_diagram(state)["version"] == "pure"
    assert state == before


def test_latest_hull_not_promoted_while_model_refresh_pending():
    from phase_agent.evidence.model_diagram_scope import decision_diagram

    state = {
        "active_model_version": "new",
        "phase_diagrams": {
            "mlip": {
                "model_version": "new",
                "version": "latest",
                "status": "completed",
                "energy_policy": "latest_valid_per_structure_v1",
                "entries": [{"source_version": "old"}],
            }
        },
        "model_refresh": {"new_model_version": "new", "status": "waiting_results"},
    }
    assert decision_diagram(state)["status"] == "pending_refresh"


def test_cached_decision_catalog_includes_valid_current_model_hull(monkeypatch):
    from phase_agent.tools.state.state_manager import agent_state_summary

    state = {
        "active_model_version": "new",
        "current_state_snapshot": {"decision_context": {"current_phase_diagram": {"mlip": {"version": "mixed"}}}},
        "phase_diagrams": {"mlip": {"version": "mixed", "model_version": "new", "entries": [{"source_version": "old"}]}},
        "phase_diagrams_by_model": {"new": {"version": "pure", "status": "completed", "model_version": "new", "entries": [{"source_version": "new"}]}},
    }
    summary = agent_state_summary(state)
    assert "phase_diagram:mlip_active_model:pure" in summary["decision_context"]["evidence_catalog"]["references"]
    state["phase_diagrams_by_model"]["new"]["entries"].append({"source_version": "old"})
    import pytest
    with pytest.raises(ValueError, match="其他模型"):
        agent_state_summary(state)
