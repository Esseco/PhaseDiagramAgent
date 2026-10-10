from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.decisions.qbc_selection.recommend_post_mc_dft import recommend_post_mc_dft
from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs


def test_budget_failure_does_not_convert_original_action():
    config = default_layered_search_config()
    config["budgets"]["total_relative_cost"] = 10000
    state = {"confirmed_config_version": "v", "qbc_candidates": [
        {"candidate_id": "s", "phase": "O3", "atom_count": 80, "x_Na_per_O2": .5,
         "predicted_Ehull": .01, "hull_impact": 1}]}
    action = {"tool": "select_dft_candidates", "task_key": "draft", "parameters": {
        "decisions": [{"candidate_id": "s", "action": "DFT_RELAX", "reason": "geometry check"}]}}
    preview, error = preview_dft_inputs(action, state, config)
    assert preview is None and "dft_budget" in error
    assert action["parameters"]["decisions"][0]["action"] == "DFT_RELAX"


def test_exhausted_budget_does_not_offer_zero_cost_plan():
    config = default_layered_search_config()
    state = {"budget_usage": {"total_relative_cost": config["budgets"]["total_relative_cost"]},
             "qbc_candidates": [{"candidate_id": "s", "atom_count": 40}]}
    assert recommend_post_mc_dft(state["qbc_candidates"], config, state)["selected_candidates"] == []
