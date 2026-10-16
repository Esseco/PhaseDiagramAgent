from phase_agent.tools.local.training_agent_review import review_training_choice, REVIEW_POLICY


def test_old_pending_review_is_preserved_without_another_agent_call():
    old = {
        "choice": "other",
        "reason": "saved review",
        "alternatives": "saved alternative",
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
        "remote_finetune_jobs": {"job": {"training_handoff": handoff.copy()}},
    }
    calls = []
    updated, waits = review_training_choice(
        state, [handoff], lambda p: calls.append(p), user_message="继续"
    )
    assert not calls
    assert updated["active_model_version"] == "old"
    assert updated["candidate_models"]["new"]["agent_review"] == old
    assert waits[0]["direction_proposal"]["direction"] == "other"


def test_approved_review_keeps_exact_saved_plan():
    review = {
        "choice": "supplement_dft",
        "reason": "补覆盖",
        "alternatives": "搜索后延",
        "followup_action": {"tool": "select_dft_candidates", "target_ids": ["s1"], "budget": 1},
    }
    handoff = {
        "stage": "awaiting_direction_approval",
        "direction_status": "approved",
        "candidate_model_version": "new",
        "evidence_type": "grouped_cross_validation_review",
    }
    state = {"candidate_models": {"new": {"validation": {}, "agent_review": review}}}

    def forbidden(payload):
        raise AssertionError("approved plan must not be rejudged")

    _, waits = review_training_choice(state, [handoff], forbidden)
    assert waits[0]["direction_proposal"]["plan"] == review["followup_action"]
