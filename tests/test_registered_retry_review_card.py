from phase_agent.runtime.review_presentation import review_card


def test_approval_displays_all_registered_retry_inputs_without_new_reservation():
    action = {"tool": "select_dft_candidates", "parameters": {
        "prepare_inputs_after_approval": True,
        "decisions": [{"candidate_id": f"S{i}", "action": "DFT_SINGLE_POINT"} for i in range(5)],
        "registered_retry_input_ids": [f"R{i}" for i in range(7)]}}
    card = review_card("p", {"agent_proposal": {"raw_action": action}})
    assert "7 个已登记重试" in card["scope"]
    assert "合计 12 个输入" in card["scope"]
    assert card["target_count"] == 5
    assert "重试预算不重复预留" in card["scope"]
