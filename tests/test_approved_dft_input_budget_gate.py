from copy import deepcopy
from phase_agent.agents.strategic.round_budget_review import round_budget_review_errors


def test_only_exact_human_accepted_registered_input_program_skips_new_route_review():
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_dft_inputs",
        "parameters": {"mode": "dft_inputs"},
        "target_ids": ["a", "b"],
        "_simulation_program": {
            "protocol": "registered_dft_inputs/v1",
            "status": "passed",
            "target_ids": ["a", "b"],
        },
        "_decision_team": {
            "backend": "codex_native",
            "revision": 1,
            "human_review": [{"kind": "human_accept", "revision": 1}],
        },
    }
    context = {"round_budget_evidence": {"required": True}}
    assert round_budget_review_errors(action, context) == []
    for change in ("no_accept", "targets", "mode", "program"):
        altered = deepcopy(action)
        if change == "no_accept":
            altered["_decision_team"]["human_review"] = []
        if change == "targets":
            altered["target_ids"] = ["a", "c"]
        if change == "mode":
            altered["parameters"]["mode"] = "other"
        if change == "program":
            altered["_simulation_program"]["status"] = "blocked"
        assert round_budget_review_errors(altered, context)
