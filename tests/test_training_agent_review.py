from phase_agent.tools.local.training_agent_review import review_training_choice


def test_agent_choice_is_cached_and_does_not_activate():
    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "evidence_type": "grouped_cross_validation_review",
    }
    state = {
        "active_model_version": "old",
        "candidate_models": {
            "new": {"validation": {}, "old_model_version": "old", "status": "validated_candidate"}
        },
        "remote_finetune_jobs": {"j": {"training_handoff": handoff.copy()}},
    }
    calls = []

    def agent(payload):
        calls.append(payload)
        return {
            "parameters": {"choice": "activate", "alternatives": "DFT可在刷新后补"},
            "reason": "误差改善",
        }

    updated, waits = review_training_choice(state, [handoff], agent)
    assert updated["active_model_version"] == "old"
    assert "回复‘同意’" in waits[0]["reason"]
    assert updated["candidate_models"]["new"]["agent_review"]["choice"] == "activate"
    review_training_choice(updated, waits, agent)
    assert len(calls) == 1
    assert calls[0]["decision_kind"] == "model_update"
    from phase_agent.decisions.agent.prepare_llm_request import needs_deep_reasoning

    assert needs_deep_reasoning(calls[0])


def test_repairs_format_without_replacing_agent_choice():
    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "evidence_type": "grouped_cross_validation_review",
    }
    state = {"active_model_version": "old", "candidate_models": {"new": {"validation": {}}}}
    responses = iter(
        [
            {"reason": "需要补DFT"},
            {
                "choice": "supplement_dft",
                "reason": "需要补DFT",
                "alternatives": "覆盖不足",
                "followup_action": {
                    "tool": "select_dft_candidates",
                    "parameters": {},
                    "target_ids": ["known"],
                    "budget": 1.0,
                    "reason": "补覆盖",
                    "expected_purpose": "验证",
                },
            },
        ]
    )
    updated, waits = review_training_choice(state, [handoff], lambda payload: next(responses))
    assert updated["candidate_models"]["new"]["agent_review"]["choice"] == "supplement_dft"
    assert waits[0]["stage"] == "execution_plan_ready"
    assert updated["active_model_version"] == "old"
