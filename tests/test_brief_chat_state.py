from phase_agent.runtime.agent_api import brief_chat_state


def test_pending_dft_state():
    assert brief_chat_state({"pending_execution_policies": {"p": {"agent_proposal": {
        "recommended_action": "select_dft_candidates"}}}}) == "当前：补充 DFT 决策待确认。"


def test_active_counts():
    text = brief_chat_state({"tasks": [{"stage": "deep_search", "status": "pending"}]})
    assert "MC 1 个待完成" in text


def test_configuration_state():
    assert "配置修订" in brief_chat_state({}, configuring=True)
