from langchain_core.messages import HumanMessage

from phase_agent.graphs.studio_chat_graph import user_text


def test_studio_accepts_input_text_blocks():
    message = HumanMessage(content=[{"type": "input_text", "text": "继续"}])
    assert user_text([message]) == "继续"
