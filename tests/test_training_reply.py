from phase_agent.runtime.training_reply import concise_training_reply


def test_legacy_prompt_never_requests_long_command():
    result = concise_training_reply({"stage": "awaiting_activation_approval", "candidate_model_version": "remote-long"}, {})
    assert "回复‘继续’" in result
    assert "remote-long" not in result
    assert "原因：" not in result


def test_agent_review_has_simple_execution_approval():
    result = concise_training_reply({"stage": "awaiting_activation_approval", "candidate_model_version": "new"},
        {"candidate_models": {"new": {"agent_review": {"choice": "activate", "reason": "误差改善"}}}})
    assert "回复‘同意’" in result
    assert "Agent建议" in result
