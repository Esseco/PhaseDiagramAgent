import pytest
from phase_agent.runtime.chat_application import RunWorkflowChatHandler, OpenWebUIRequestError
from phase_agent.runtime.studio_session_scope import studio_project_session


def test_studio_switches_thread_without_reusing_dialogue(tmp_path):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    handler.conversation_id = "first"
    handler.recent_dialogue = [{"role": "user", "content": "first private context"}]
    with studio_project_session():
        reply = handler([{"role": "user", "content": "现在什么阶段"}], conversation_id="second")
    assert "当前" in reply
    assert all(row.get("content") != "first private context" for row in handler.recent_dialogue)


def test_legacy_http_single_session_guard_remains(tmp_path):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    handler.conversation_id = "first"
    with pytest.raises(OpenWebUIRequestError, match="另一个"):
        handler([{"role": "user", "content": "现在什么阶段"}], conversation_id="second")


def test_each_studio_thread_restores_its_own_context(tmp_path, monkeypatch):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    received = []

    def respond(messages, *, conversation_id=None):
        received.append((conversation_id, [row["content"] for row in messages]))
        return "reply " + messages[-1]["content"]

    monkeypatch.setattr(handler, "_respond", respond)
    with studio_project_session():
        handler([{"role": "user", "content": "A chemistry"}], conversation_id="A")
        handler([{"role": "user", "content": "B question"}], conversation_id="B")
        handler([{"role": "user", "content": "A followup"}], conversation_id="A")
    assert received[0][1] == ["A chemistry"]
    assert received[1][1] == ["B question"]
    assert received[2][1] == ["A chemistry", "reply A chemistry", "A followup"]
    assert handler.state_path == tmp_path / "state.json"


def test_checkpointed_thread_context_is_available_to_fresh_handler(tmp_path, monkeypatch):
    handler = RunWorkflowChatHandler({"state_path": str(tmp_path / "state.json")})
    seen = []
    monkeypatch.setattr(
        handler, "_respond", lambda messages, **kwargs: seen.extend(messages) or "ok"
    )
    history = [
        {"role": "user", "content": "saved question"},
        {"role": "assistant", "content": "saved answer"},
        {"role": "user", "content": "continue"},
    ]
    with studio_project_session(history), studio_project_session():
        handler([history[-1]], conversation_id="restored")
    assert seen == history


def test_fresh_studio_handler_restores_only_last_turn(tmp_path, monkeypatch):
    from phase_agent.persistence.memory.conversation import remember_turn

    path = tmp_path / "state.json"
    remember_turn(path, "A", "旧内容", "旧答复")
    correction = "背景" * 600 + "接着上次计划，不重新选方向"
    remember_turn(path, "A", correction, "上次答复")
    handler = RunWorkflowChatHandler({"state_path": str(path)})
    seen = []
    monkeypatch.setattr(
        handler, "_respond", lambda messages, **kwargs: seen.extend(messages) or "ok"
    )
    with studio_project_session():
        handler([{"role": "user", "content": "继续"}], conversation_id="A")
    assert seen == [
        {"role": "user", "content": correction},
        {"role": "assistant", "content": "上次答复"},
        {"role": "user", "content": "继续"},
    ]
    seen.clear()
    with studio_project_session():
        handler([{"role": "user", "content": "继续"}], conversation_id="B")
    assert seen == [{"role": "user", "content": "继续"}]


def test_continue_after_restart_reuses_pending_plan_without_model(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import read_json
    from copy import deepcopy

    calls = []

    def model(payload):
        calls.append(payload)
        return {
            "tool": "check_convergence",
            "parameters": {},
            "budget": 0,
            "reason": "核对现有数据",
        }

    first = make_handler(tmp_path, model, {"check_convergence": lambda **kw: {"converged": False}})
    first([{"role": "user", "content": "建议下一步"}], conversation_id="A")
    before = deepcopy(read_json(first.state_path, {})["pending_execution_policies"])
    restarted = make_handler(
        tmp_path, lambda payload: (_ for _ in ()).throw(AssertionError("不能重新思考")), {}
    )
    reply = restarted([{"role": "user", "content": "继续"}], conversation_id="A")
    assert "同意" in reply
    assert read_json(first.state_path, {})["pending_execution_policies"] == before
    assert len(calls) == 1


def test_continue_cannot_reoffer_delivered_approval_without_execution_receipt(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.graphs.approval_graph import durable_approval
    from phase_agent.tools.policy.file_approval import proposal_hash

    first = make_handler(
        tmp_path,
        lambda p: {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "核对"},
        {},
    )
    first([{"role": "user", "content": "建议"}], conversation_id="A")
    state = read_json(first.state_path, {})
    invocation, row = next(iter(state["pending_execution_policies"].items()))
    durable_approval(
        first.state_path,
        {
            "invocation_id": invocation,
            "config_version": state.get("confirmed_config_version"),
            "proposal_hash": proposal_hash(row["agent_proposal"]),
            "revision": row.get("revision", 0),
        },
        "approve",
    )
    restarted = make_handler(
        tmp_path, lambda p: (_ for _ in ()).throw(AssertionError("不能重想")), {}
    )
    reply = restarted([{"role": "user", "content": "继续"}], conversation_id="A")
    assert "已交付" in reply and "尚需核对" in reply
