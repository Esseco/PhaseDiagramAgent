from phase_agent.tools.local.training_agent_review import review_training_choice, REVIEW_POLICY


def test_old_pending_review_rejudged_and_activation_stage_restored():
    old = {
        "choice": "other",
        "reason": "补验证配置",
        "alternatives": "等待",
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

    def agent(payload):
        calls.append(payload)
        assert "边际收益" in payload["instruction"]
        assert "两条路线" in payload["instruction"]
        return {
            "parameters": {"choice": "activate", "alternatives": "暂无明确高收益DFT目标"},
            "reason": "先刷新并搜索新区域",
        }

    updated, waits = review_training_choice(state, [handoff], agent, user_message="继续")
    assert len(calls) == 1
    assert waits[0]["stage"] == "awaiting_activation_approval"
    assert updated["active_model_version"] == "old"
    assert updated["candidate_models"]["new"]["agent_review"]["review_policy"] == REVIEW_POLICY
    assert updated["candidate_models"]["new"]["superseded_agent_reviews"] == [old]
    review_training_choice(updated, waits, agent, user_message="继续")
    assert len(calls) == 1


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
