import pytest

from phase_agent.analysis.phase.mc_snapshot import mc_model_snapshot
from phase_agent.tools.relax_mc_batch.plan import _latest_diagram_history
from tests.test_relax_mc_latest_policy import sample


def current_sample():
    state, diagram = sample()
    state["phase_records"] = [state["phase_records"][1]]
    diagram["entries"] = [diagram["entries"][1]]
    diagram["entries"][0]["phase"] = "O3"
    diagram.update(energy_policy="current_model_only_v1", mixed_model_versions=False,
                   source_model_versions=["m1"])
    state["model_refresh"] = {"hull_source_policy": "current_model_only_v1",
                              "new_model_version": "m1"}
    return state, diagram


def test_current_only_hull_used_for_mc_and_frozen_history():
    state, diagram = current_sample()
    assert mc_model_snapshot(state, "m1") is diagram
    assert [r["record_id"] for r in _latest_diagram_history(state, diagram, "m1")] == ["new"]


@pytest.mark.parametrize("mutation", ["unapproved", "old_source", "mixed", "no_entries", "wrong_model"])
def test_current_only_hull_rejects_wrong_binding(mutation):
    state, diagram = current_sample()
    if mutation == "unapproved":
        state["model_refresh"]["hull_source_policy"] = "latest_valid_per_structure_v1"
    elif mutation == "old_source":
        diagram["entries"][0]["source_version"] = "m0"
    elif mutation == "mixed":
        diagram["mixed_model_versions"] = True
    elif mutation == "no_entries":
        diagram["entries"] = []
    else:
        state["model_refresh"]["new_model_version"] = "m0"
    with pytest.raises(ValueError):
        mc_model_snapshot(state, "m1")
    with pytest.raises(ValueError):
        _latest_diagram_history(state, diagram, "m1")


def test_current_only_hull_rejects_missing_model_round():
    state, diagram = current_sample()
    del state["upload_layout"]["model_rounds"]["m1"]
    with pytest.raises(ValueError):
        mc_model_snapshot(state, "m1")
    with pytest.raises(ValueError):
        _latest_diagram_history(state, diagram, "m1")
