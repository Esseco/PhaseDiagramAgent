from copy import deepcopy
import pytest
from phase_agent.analysis.phase.energy_consistency import (
    legacy_mc_mean_record_ids,
    assert_snapshot_energy_consistency,
)
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions
from phase_agent.tools.workflows.prepare_mc_preview import prepare_mc_preview


def _state():
    return {
        "phase_diagrams": {"mlip": {"entries": [{"record_id": "r"}]}},
        "phase_records": [{"record_id": "r", "source_task_id": "m", "energy": -20.0}],
        "tasks": [
            {
                "task_id": "m",
                "stage": "deep_search",
                "outputs": {
                    "energy": -20.0,
                    "energy_per_atom": -5.0,
                    "qbc": {"member_count": 4},
                    "pool_summary": [{"energy_mean_per_atom": -5.0}],
                },
            }
        ],
    }


def test_proven_mean_hull_blocks_dft_and_mc_planning():
    state = _state()
    assert legacy_mc_mean_record_ids(state, state["phase_diagrams"]["mlip"]) == ["r"]
    result = validate_dft_agent_decisions(
        {"decisions": []}, [], state, config={}, config_version="c", remaining_budget=0
    )
    assert result["valid"] is False and "口径不一致" in result["errors"][0]
    with pytest.raises(ValueError, match="口径不一致"):
        prepare_mc_preview({"tool": "allocate_mc_bohb"}, state, {}, {})
    from phase_agent.tools.relax_mc_batch.plan import attach_preview

    with pytest.raises(ValueError, match="口径不一致"):
        attach_preview({"parameters": {"mode": "relax_mc_inputs"}}, state, {}, {})


@pytest.mark.parametrize(
    "change",
    ["main_source", "single", "predict", "other_record", "different_energy", "unknown_pool"],
)
def test_guard_does_not_equate_committee_evidence_with_mean_energy(change):
    state = deepcopy(_state())
    task = state["tasks"][0]
    out = task["outputs"]
    if change == "main_source":
        out["energy_source"] = "main_model_selected_frame"
    if change == "single":
        out["qbc"]["member_count"] = 1
    if change == "predict":
        task["stage"] = "relax_and_feature"
    if change == "other_record":
        state["phase_diagrams"]["mlip"]["entries"][0]["record_id"] = "other"
    if change == "different_energy":
        out["energy_per_atom"] = -5.1
    if change == "unknown_pool":
        out["pool_summary"] = []
    assert_snapshot_energy_consistency(state, state["phase_diagrams"]["mlip"])
