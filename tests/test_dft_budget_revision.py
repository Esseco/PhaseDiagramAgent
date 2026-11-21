from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.agents.reliability.selection.recommend_post_mc_dft import recommend_post_mc_dft
from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs


def test_legacy_cost_limit_does_not_convert_original_action(tmp_path):
    config = default_layered_search_config()
    config["budgets"]["enforcement"] = "hard"
    config["budgets"]["total_relative_cost"] = 10000
    state = {
        "confirmed_config_version": "v",
        "qbc_candidates": [
            {
                "candidate_id": "s",
                "phase": "O3",
                "atom_count": 80,
                "x_Na_per_O2": 0.5,
                "predicted_Ehull": 0.01,
                "hull_impact": 1,
                "geometry_optimization_evidence": {
                    "source_task_id": "prior-dft-s",
                    "residual_force_ev_per_angstrom": 0.2,
                    "force_tolerance_ev_per_angstrom": 0.05,
                },
            }
        ],
    }
    from tests.test_dft_input_preview import add_monitor_evidence

    add_monitor_evidence(state, tmp_path)
    action = {
        "tool": "select_dft_candidates",
        "task_key": "draft",
        "parameters": {
            "decisions": [{"candidate_id": "s", "action": "DFT_RELAX", "reason": "geometry check"}]
        },
    }
    preview, error = preview_dft_inputs(action, state, config)
    assert error is None and preview["task_count"] == 1
    assert action["parameters"]["decisions"][0]["action"] == "DFT_RELAX"


def test_legacy_cost_exhaustion_does_not_block_dft_selection():
    config = default_layered_search_config()
    config["budgets"]["enforcement"] = "hard"
    config["budgets"]["total_relative_cost"] = 0
    state = {
        "budget_usage": {"total_relative_cost": 0},
        "qbc_candidates": [{"candidate_id": "s", "atom_count": 40}],
    }
    selected = recommend_post_mc_dft(state["qbc_candidates"], config, state)["selected_candidates"]
    assert [row["candidate_id"] for row in selected] == ["s"]
