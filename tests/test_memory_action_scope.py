from phase_agent.analysis.state.memory_action_scope import memory_action_scope


def test_mixed_stages_are_not_inferred_from_last_task():
    tasks = [{"stage": "relax_and_feature", "status": "completed"},
             {"stage": "dft_single_point", "status": "completed"}]
    scope = memory_action_scope({"tasks": tasks})
    assert scope["action"] is None
    assert scope == memory_action_scope({"tasks": tasks[::-1]})


def test_explicit_decision_scope_overrides_historical_mixture():
    scope = memory_action_scope({"decision_action": "select_dft_candidates",
                                "tasks": [{"stage": "relax_and_feature"}]})
    assert scope["source"] == "explicit_decision_action"
    assert scope["action"] == "select_dft_candidates"


def test_empty_or_unknown_tasks_do_not_guess():
    assert memory_action_scope({})["action"] is None
    assert memory_action_scope({"tasks": [{}]})["action"] is None
