"""Complete conversation sequences protect proposals and execution boundaries."""

from copy import deepcopy
import json
import pytest
from tests.test_unified_react_dialogue import make_handler
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.decisions.agent.dialogue_contract import dialogue_errors, dialogue_output_contract


def test_complete_conversation_view_discuss_reject_restart_approve(tmp_path):
    calls, executed = [], []
    action = {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "核对现有数据"}
    outcomes = [
        action,
        {"kind": "answer", "answer": "解释方案"},
        {"kind": "inspect_configuration", "answer": "查看当前参数"},
        {"kind": "answer", "answer": "暂不批准，继续讨论"},
        {"kind": "answer", "answer": "重启后仍能理解前文"},
        action,
    ]

    def model(payload):
        calls.append(payload)
        return deepcopy(outcomes[len(calls) - 1])

    handler = make_handler(
        tmp_path,
        model,
        {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}},
    )
    handler.config_revision_factory = lambda state: pytest.fail(
        "Viewing settings cannot create a revision"
    )
    handler([{"role": "user", "content": "分析并建议下一步"}], conversation_id="a")
    pending = deepcopy(read_json(handler.state_path, {})["pending_execution_policies"])
    assert handler([{"role": "user", "content": "为何这么安排"}], conversation_id="a") == "解释方案"
    viewed = handler([{"role": "user", "content": "先回去让我看看参数"}], conversation_id="a")
    assert "本次仅查看" in viewed and handler.config_delegate is None
    assert read_json(handler.state_path, {})["pending_execution_policies"] == pending
    assert (
        handler([{"role": "user", "content": "先不要执行，我们讨论一下"}], conversation_id="a")
        == "暂不批准，继续讨论"
    )
    assert not executed
    handler([{"role": "user", "content": "拒绝"}], conversation_id="a")
    assert not read_json(handler.state_path, {}).get("pending_execution_policies")
    restarted = make_handler(
        tmp_path,
        model,
        {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}},
    )
    assert "重启后" in restarted(
        [{"role": "user", "content": "还记得我们在讨论什么吗"}], conversation_id="a"
    )
    assert calls[-1]["conversation_facts"]["recent_turns"]
    restarted([{"role": "user", "content": "重新给出建议"}], conversation_id="a")
    calls_before = len(calls)
    restarted([{"role": "user", "content": "同意"}], conversation_id="a")
    restarted([{"role": "user", "content": "同意"}], conversation_id="a")
    assert len(calls) == calls_before and len(executed) == 1


def test_typed_contract_rejects_mixed_or_extra_executable_fields():
    assert not dialogue_errors({"kind": "inspect_configuration", "answer": "查看"}, True)
    assert dialogue_errors(
        {"kind": "answer", "answer": "已执行", "tool": "generate_branches"}, True
    )
    assert dialogue_errors({"kind": "answer", "answer": "查看", "submitted": True}, True)
    assert dialogue_errors({"kind": "answer", "answer": "   "}, True)
    assert dialogue_output_contract()["schema"]["additionalProperties"] is False


def test_node_and_model_timings_are_persisted_and_rendered(tmp_path):
    from phase_agent.runtime.turn_process import record_turn, timed_call, render_process_panel

    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(json.dumps({"state_path": "state.json"}), encoding="utf-8")
    with record_turn(runtime, "问一个问题", "a"):
        assert timed_call("model", lambda: "答复", category="model") == "答复"
        with pytest.raises(ValueError):
            timed_call("failed_node", lambda: (_ for _ in ()).throw(ValueError("test failure")))
    record = json.loads(
        next((tmp_path / "workflow_state/dialogue_process").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert record["duration_seconds"] >= 0
    assert len(record["timings"]) == 2
    assert record["timings"][1]["status"] == "failed"
    assert "模型调用：1 次" in render_process_panel(runtime)


def test_read_only_answer_and_view_do_not_enter_scientific_lifecycle(tmp_path):
    outcomes = [
        {"kind": "answer", "answer": "依据已保存的事实答复"},
        {"kind": "inspect_configuration", "answer": "查看参数"},
    ]
    handler = make_handler(tmp_path, lambda payload: outcomes.pop(0))
    handler._run = lambda *args, **kw: pytest.fail(
        "Read-only conversation must not collect, analyze or prepare scientific work"
    )
    assert (
        handler([{"role": "user", "content": "解释现状"}], conversation_id="a")
        == "依据已保存的事实答复"
    )
    assert "本次仅查看" in handler(
        [{"role": "user", "content": "参数给我看看"}], conversation_id="a"
    )


@pytest.mark.parametrize("changed", [False, True])
def test_scientific_proposal_reuse_or_refresh_preserves_usage(tmp_path, changed):
    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts

    calls = []

    def model(payload):
        calls.append(deepcopy(payload))
        return {
            "tool": "check_convergence",
            "parameters": {},
            "budget": 0,
            "reason": "review evidence",
            "_llm_usage": {"calls": 1, "input_tokens": 7},
        }

    handler = make_handler(tmp_path, model)

    def scientific():
        payload = deepcopy(calls[0])
        if changed:
            payload["state"]["current_convex_hull"]["dft"]["version"] = "new-result"
        return handler.workflow_kwargs["agent_client"](payload)

    result = decide_from_saved_facts(handler, {}, "suggest next step", scientific)
    assert len(calls) == (2 if changed else 1)
    persisted = read_json(handler.state_path, {})["budget_usage"]["llm"]
    unrecorded = result.get("_llm_usage") or {}
    assert persisted["calls"] + unrecorded.get("calls", 0) == len(calls)
    assert persisted["input_tokens"] + unrecorded.get("input_tokens", 0) == 7 * len(calls)
    assert handler.workflow_kwargs["agent_client"] is model


def test_current_message_is_separate_from_historical_context():
    from unittest.mock import patch
    from tests.test_deepseek_client import FakeResponse, completion
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client

    client = create_deepseek_client(api_key="test", thinking="disabled")
    with patch(
        "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
        return_value=FakeResponse(completion('{"kind":"answer","answer":"ok"}')),
    ) as request:
        client(
            {
                "unified_dialogue": True,
                "user_instruction": "now propose a plan",
                "conversation_facts": {"recent_turns": ["previously only explain"]},
            }
        )
    messages = json.loads(request.call_args.args[0].data)["messages"]
    assert messages[-1]["role"] == "user"
    assert "now propose a plan" in messages[-1]["content"]
    assert "previously only explain" not in messages[-1]["content"]
    assert json.loads(messages[1]["content"])["conversation_facts"]["recent_turns"]
