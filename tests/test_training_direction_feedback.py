from phase_agent.tools.local.training_agent_review import review_training_choice


def test_correction_revises_cached_direction_and_is_not_approval():
    old = {
        "choice": "other",
        "reason": "check interface",
        "alternatives": "wait",
        "followup_action": {"tool": "adjust_strategy"},
    }
    handoff = {
        "stage": "awaiting_direction_approval",
        "candidate_model_version": "new",
        "evidence_type": "grouped_cross_validation_review",
    }
    state = {
        "active_model_version": "old",
        "candidate_models": {"new": {"validation": {}, "agent_review": old}},
        "remote_finetune_jobs": {"j": {"training_handoff": handoff.copy()}},
    }
    calls = []

    def agent(payload):
        calls.append(payload)
        assert payload["decision_context"]["user_feedback"] == "不是接口错误，就是模型的误差"
        assert payload["decision_context"]["previous_review"]["reason"] == "check interface"
        return {
            "parameters": {"choice": "activate", "alternatives": "compare after refresh"},
            "reason": "updated reasoning",
        }

    state, waits = review_training_choice(
        state, [handoff], agent, user_message="不是接口错误，就是模型的误差"
    )
    assert len(calls) == 1
    assert state["active_model_version"] == "old"
    assert state["candidate_models"]["new"]["superseded_agent_reviews"] == [old]
    assert waits[0]["direction_proposal"]["user_feedback"] == "不是接口错误，就是模型的误差"
    review_training_choice(state, waits, agent, user_message="继续")
    assert len(calls) == 1
