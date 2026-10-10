from phase_agent.decisions.agent.post_dft_review import valid_post_dft_review
from phase_agent.decisions.agent.propose_tool_action import propose_agent_tool_action
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from tests.test_post_dft_review import review, parameters


def test_supplement_can_support_needed_finetuning():
    fields = review()
    fields.update(choice="supplement_dft", finetune_recommendation="now")
    assert valid_post_dft_review({"tool": "select_dft_candidates", "post_dft_review": fields,
        "parameters": {"decisions": [{"candidate_id": "S-1", "action": "DFT_SINGLE_POINT"}]}})


def test_pause_requires_explicit_stop_basis_and_full_comparison():
    fields = review()
    fields["choice"] = "stop"
    action = {"tool": "pause_search", "post_dft_review": fields}
    assert not valid_post_dft_review(action)
    fields["stop_status"] = "budget_stop"
    assert valid_post_dft_review(action)
    fields.pop("convergence_assessment")
    assert not valid_post_dft_review(action)


def test_parameter_failure_preserves_complete_analysis_without_authorizing():
    def client(payload):
        params = parameters()
        params["generation_plan"][0]["quota"] = "ten"
        return {"tool": "generate_branches", "parameters": params, "post_dft_review": review()}
    result = propose_agent_tool_action({"available_branches": [{}],
        "decision_context": {"post_dft_assessment": {"status": "evaluated"}}},
        allowed_tools=["generate_branches", "pause_search"], agent_client=client)
    assert result["decision_source"] == "rule"
    assert result["failed_post_dft_review"]["choice"] == "search"
    text = format_workflow_reply({"status": "not_configured", "post_dft_review": result["failed_post_dft_review"],
                                 "reason": result["fallback_reason"]}, "state.json")
    assert "补DFT取舍" in text and "收敛与停止" in text
    assert "回复“同意”" not in text
