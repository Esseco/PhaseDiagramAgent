"""长期人工建议、近期观测和离线审批恢复的跨层验证。"""

import json

from analysis_layer.state.build_decision_context import build_decision_context
from execution_layer.workflows.run_event_loop import run_event_loop
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from tests.test_event_loop import _session
from decision_layer.strategy.propose_round_strategy import propose_round_strategy
from decision_layer.qbc_selection.decide_dft_actions import decide_dft_actions
from execution_layer.state.state_manager import agent_state_summary, update_state_snapshot


def test_long_term_advice_revision_survives_restart_and_needs_new_approval(tmp_path):
    path = tmp_path / "state.json"
    approvals = tmp_path / "approvals"
    calls, prompts = [], []
    registry = create_tool_registry({"check_convergence": lambda **kw: calls.append(kw) or {"status": "completed"}})

    def agent(payload):
        prompts.append(payload)
        return {"tool": "check_convergence", "budget": 0, "parameters": {}, "reason": "检查"}

    kwargs = dict(registry=registry, agent_client=agent, execution_mode="interactive", state_path=path, approval_directory=approvals)
    first = run_event_loop({}, _session(), initial_long_term_advice=["优先 DFT-SP"], **kwargs)
    assert prompts[-1]["decision_context"]["long_term_human_advice"]["items"] == ["优先 DFT-SP"]
    folder = next(approvals.iterdir())
    decision_path = folder / "decision-r000.json"
    feedback = json.loads(decision_path.read_text(encoding="utf-8"))
    feedback.update(comment="同意", long_term_advice=["优先 DFT-SP", "Relax 必须解释理由"])
    decision_path.write_text(json.dumps(feedback, ensure_ascii=False), encoding="utf-8")
    revised = run_event_loop(path, _session(), **kwargs)
    assert revised["status"] == "awaiting_approval"
    assert calls == []
    assert prompts[-1]["decision_context"]["long_term_human_advice"]["items"][-1] == "Relax 必须解释理由"
    approval_path = folder / "decision-r001.json"
    feedback = json.loads(approval_path.read_text(encoding="utf-8"))
    feedback["comment"] = "同意"
    approval_path.write_text(json.dumps(feedback, ensure_ascii=False), encoding="utf-8")
    finished = run_event_loop(path, _session(), **kwargs)
    assert len(calls) == 1
    assert finished["state"]["decision_memory"]["version"] == 2
    # 下一 action 自动读取已保存建议，初始化参数不能覆盖人工修订。
    run_event_loop(path, _session(), initial_long_term_advice=["过时初始化值"], **kwargs)
    assert prompts[-1]["decision_context"]["long_term_human_advice"]["items"][0] == "优先 DFT-SP"


def test_recent_observations_are_bounded_and_not_promoted_to_advice():
    state = {
        "rewards": [{"batch_id": str(i), "reward": i, "energy_method": "mlip"} for i in range(9)],
        "action_records": [{"record_id": "a1", "feedback_history": [{"comment": "本次预算减半"}]}],
        "phase_diagrams": {"dft": {"version": "h1", "entries": [{"record_id": "S1", "ehull": 0, "is_stable": True}]}},
    }
    context = build_decision_context(state)
    assert context["long_term_human_advice"]["items"] == []
    assert len(context["recent_experience"]["rewards"]) == 5
    assert context["recent_experience"]["rewards"][0]["comparison_status"].startswith("unversioned")
    assert context["current_phase_diagram"]["dft"]["stable_count"] == 1
    assert context["recent_experience"]["actions"][0]["recent_comments"][0]["comment"] == "本次预算减半"


def test_round_and_dft_agents_receive_separate_memory_context():
    state = {"decision_memory": {"version": 1, "long_term_advice": ["优先 SP"]}}
    payloads = []
    def agent(payload):
        payloads.append(payload)
        return {"decisions": []}
    snapshot = agent_state_summary(update_state_snapshot(state))
    propose_round_strategy(snapshot, agent_client=agent, config={})
    decide_dft_actions([], snapshot, agent_client=agent)
    assert len(payloads) == 2
    assert all(p["decision_context"]["long_term_human_advice"]["items"] == ["优先 SP"] for p in payloads)
