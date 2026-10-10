from phase_agent.runtime.decision_visibility import decision_summary, render_decision_card


def test_projection_preserves_declared_tradeoffs_without_inventing():
    action = {"tool": "select_dft_candidates", "reason": "近凸包缺少DFT", "post_dft_review": {
        "finetune_assessment": "暂缓微调"}, "evidence_refs": ["report-1"]}
    result = decision_summary(action)
    assert result["recommendation"] == "补充 DFT"
    assert result["assessments"] == {"微调取舍": "暂缓微调"}
    assert result["alternatives"] == []


def test_card_is_short_escaped_and_details_collapsed():
    state = {"pending_execution_policies": {"p": {"agent_proposal": {"raw_action": {
        "tool": "generate_branches", "reason": "<script>bad</script>",
        "evidence_refs": ["report-1"]}}}}}
    text = render_decision_card(state)
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "待批准，尚未执行" in text
    assert "<details>" in text and "<details open" not in text
    assert "暂无唯一" in render_decision_card({})
