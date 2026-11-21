from copy import deepcopy
from phase_agent.analysis.cost.estimate_proposal_cost import estimate_proposal_cost
from phase_agent.runtime.review_presentation import review_card


def test_frozen_dft_selection_preview_cost_used_for_new_and_legacy_cards():
    preview = {
        "relative_cost": 964.75,
        "task_count": 2,
        "workload": [{"stage": "dft_single_point", "tasks": 2, "initial_cost": 964.75}],
    }
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "a", "action": "DFT_SINGLE_POINT"},
                {"candidate_id": "b", "action": "DFT_SINGLE_POINT"},
            ],
            "dft_selection_preview": preview,
        },
        "budget": 964.75,
    }
    estimate = estimate_proposal_cost(action, {})
    assert estimate["estimated_total_cost"] == 964.75
    proposal = {
        "raw_action": action,
        "estimated_cost": {"estimated_total_cost": 0.0, "workload": []},
    }
    record = {"agent_proposal": proposal}
    before = deepcopy(record)
    card = review_card("plan", record)
    assert card["estimated_cost"]["estimated_total_cost"] == 964.75
    assert record == before
