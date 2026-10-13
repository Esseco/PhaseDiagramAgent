from phase_agent.science.mc.schedule_tiered_mc import schedule_tiered_mc
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)


def schedule(gap, segments):
    return schedule_tiered_mc(
        [
            {
                "branch_id": "B",
                "structure_id": "new-ground",
                "structure_path": "new-ground.vasp",
                "relaxed_ehull": gap,
                "relaxed_ehull_unit": "eV/atom",
                "ehull_source": "phase_diagram",
            }
        ],
        {"segments": segments},
        policy=default_layered_search_config()["mc_policy"],
        total_budget=1000,
        seed=42,
        model_version="new",
        hull_reference_version="new-hull",
        phase_diagram_version="new-diagram",
    )


def old_segments():
    return [
        {
            "branch_id": "B",
            "status": "completed",
            "model_version": "old",
            "energy_improvement": 0,
            "stop_reason": "unknown",
        }
        for _ in range(2)
    ]


def test_refreshed_ground_enters_mc_with_new_structure_despite_old_limit():
    result = schedule(1e-10, old_segments())
    assert len(result["actions"]) == 1
    task = result["actions"][0]
    assert task["structure_id"] == "new-ground"
    assert task["structure_path"] == "new-ground.vasp"
    assert task["segment_index"] == 0
    assert task["segment_reason"] == "refreshed_hull_ground_state"


def test_near_hull_non_ground_keeps_old_rules():
    assert schedule(0.0005, old_segments())["actions"] == []


def test_ground_cannot_ignore_current_model_segment_limit():
    current = [{**row, "model_version": "new"} for row in old_segments()]
    assert schedule(0, old_segments() + current)["actions"] == []
