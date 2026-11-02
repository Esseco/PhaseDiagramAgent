import pytest
from phase_agent.analysis.feedback.model_refresh_plan import (
    supplemental_relaxation_required,
    build_model_refresh_plan,
)


@pytest.mark.parametrize(
    "gap,force,expected",
    [
        (0.000999, 0.0, True),
        (0.001, 0.1, False),
        (0.001, 0.100001, True),
        (0.01, 0.100001, True),
        (0.010001, 0.2, False),
        (0.010001, 0.200001, True),
        (0.2, 0.21, True),
    ],
)
def test_force_boundaries(gap, force, expected):
    assert supplemental_relaxation_required(gap, [[force, 0, 0]]) is expected


def test_far_requires_force_evidence():
    with pytest.raises(ValueError):
        supplemental_relaxation_required(0.02, [])


def test_far_historical_flag_does_not_override_current_far_sampling():
    rows = [
        {
            "structure_id": str(i),
            "structure_sha256": str(i),
            "ehull": 0.02,
            "source_version": "old",
            "phase": "O3",
            "x_Na_per_O2": 0.5,
            "historical_near_hull": True,
        }
        for i in range(20)
    ]
    plan = build_model_refresh_plan(rows, old_version="old", new_version="new")
    assert len(plan["candidates"]) == 2
    assert plan["policy"]["far_force_threshold"] == 0.2


def test_completed_predict_requires_supplementary_relaxation(tmp_path):
    from tests.test_model_refresh_workflow import recovered_state
    from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

    state, _, _, _ = recovered_state(tmp_path)
    state["model_refresh"]["plan"]["policy"]["force_followup_version"] = 1
    state["tasks"][0]["outputs"]["forces_unit"] = "eV/angstrom"
    state["tasks"][0]["outputs"]["forces"] = [[0.11, 0, 0]] * 4
    new, wait = advance_refresh(state)
    assert wait is None
    assert new["model_refresh"]["status"] == "supplement_ready"
    assert len(new["model_refresh"]["supplemental_candidates"]) == 1
    assert new["model_refresh"]["supplemental_candidates"][0]["operation"] == "relax"


def test_frozen_near_hull_threshold_survives_new_hull_change(tmp_path):
    from tests.test_model_refresh_workflow import recovered_state
    from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

    state, _, _, _ = recovered_state(tmp_path)
    state["model_refresh"]["plan"]["policy"]["force_followup_version"] = 1
    state["phase_diagrams"]["mlip"]["entries"][0]["ehull"] = 0.02
    new, wait = advance_refresh(state)
    assert wait is None
    assert new["model_refresh"]["status"] == "supplement_ready"


def test_missing_force_cannot_finish_refresh(tmp_path):
    from tests.test_model_refresh_workflow import recovered_state
    from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

    state, _, _, _ = recovered_state(tmp_path)
    state["model_refresh"]["plan"]["policy"]["force_followup_version"] = 1
    state["tasks"][0]["outputs"].pop("forces")
    new, wait = advance_refresh(state)
    assert wait and "受力" in wait
    assert new["model_refresh"]["status"] == "waiting_results"
