"""Synthetic recovery regressions; no live projects, services or model calls."""

from copy import copy, deepcopy

import pytest

from phase_agent.graphs.project.runner import run_search_workflow_graph
from phase_agent.graphs.scientific_graph import scientific_graph
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_unified_react_dialogue import make_handler


def lifecycle(tmp_path, failed_stage, final_status="done"):
    calls = []
    failure = [True]

    def stage(name, frame):
        calls.append(name)
        if name == failed_stage and failure[0]:
            raise RuntimeError("synthetic interrupted stage")
        return frame

    options = dict(
        checkpoint_path=tmp_path / "lifecycle.sqlite",
        project_thread="project",
        initialize=lambda: {
            "_workflow_prepared": True,
            "snapshot": {"config_version": "v1"},
        },
        collect=lambda frame: stage("collect", frame),
        analyze=lambda frame: stage("observe", frame),
        wait=lambda frame: frame,
        assess=lambda frame: stage("assess", frame),
        act=lambda frame, **kwargs: stage("act", frame),
        finalize=lambda frame: {"status": final_status},
    )
    return options, calls, failure


@pytest.mark.parametrize("failed_stage", ["observe", "assess"])
@pytest.mark.parametrize("final_status", ["done", "awaiting_approval"])
def test_new_request_rebinds_safe_recovery_and_caches_completion(
    tmp_path, failed_stage, final_status
):
    options, calls, failure = lifecycle(tmp_path, failed_stage, final_status)
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run_search_workflow_graph(invocation_id="old", **options)
    failure[0] = False
    result = run_search_workflow_graph(invocation_id="new", **options)
    assert result == {"status": final_status}
    assert calls.count("collect") == 1  # Completed nested observations must stay completed.
    with CompactSqliteSaver.from_conn_string(str(options["checkpoint_path"])) as saver:
        graph = copy(scientific_graph())
        graph.checkpointer = saver
        snapshot = graph.get_state({"configurable": {"thread_id": "project"}})
        assert snapshot.values["request_id"] == "new"
    completed_calls = list(calls)
    assert run_search_workflow_graph(invocation_id="new", **options) == result
    assert calls == completed_calls


def test_new_request_cannot_replay_failed_execution(tmp_path):
    options, calls, failure = lifecycle(tmp_path, "act")
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run_search_workflow_graph(invocation_id="old", **options)
    previous = list(calls)
    failure[0] = False
    result = run_search_workflow_graph(invocation_id="new", **options)
    assert result["status"] == "execution_reconciliation_required"
    assert result["submitted"] is False
    assert calls == previous


def test_new_feedback_cannot_enter_an_old_inflight_request(tmp_path):
    options, calls, failure = lifecycle(tmp_path, "observe")
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run_search_workflow_graph(invocation_id="old", **options)
    failure[0] = False
    previous = list(calls)
    with pytest.raises(ValueError, match="request|invocation"):
        run_search_workflow_graph(
            invocation_id="new", human_feedback={"decision": "approve"}, **options
        )
    assert calls == previous


def test_approved_pending_plan_isolated_from_old_lifecycle_checkpoint(tmp_path):
    """A valid approval can consume its plan after an unrelated stale frame."""
    from phase_agent.tools.step_runner.file_protocol import write_json

    state_path = tmp_path / "state.json"
    write_json(
        state_path,
        {
            "execution_recovery_report": {"status": "clear", "unsettled": []},
            "pending_execution_policies": {"new": {"agent_proposal": {}}},
            "pending_execution_recoveries": [],
            "budget_reservations": {},
            "tasks": [],
        },
    )
    options, calls, failure = lifecycle(tmp_path, "observe", "awaiting_approval")
    options["business_state_path"] = state_path
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run_search_workflow_graph(invocation_id="old", **options)
    failure[0] = False
    result = run_search_workflow_graph(
        invocation_id="new",
        human_feedback={"decision": "approve"},
        **options,
    )
    assert result == {"status": "awaiting_approval"}
    assert calls[-2:] == ["assess", "act"]


def test_approval_resumes_wait_boundary_with_same_lifecycle_identity(tmp_path):
    options, calls, failure = lifecycle(tmp_path, "never", "awaiting_approval")
    first = run_search_workflow_graph(invocation_id="same", **options)
    assert first == {"status": "awaiting_approval"}
    previous = list(calls)
    result = run_search_workflow_graph(
        invocation_id="same",
        human_feedback={"decision": "approve"},
        **options,
    )
    assert result == {"status": "awaiting_approval"}
    assert calls[: len(previous)] == previous


def recovery_handler(tmp_path, monkeypatch, outcome, *, partial_commit=False):
    from phase_agent.tools.workflows import recover_saved_results as recovery

    requests = []

    def model(payload):
        requests.append(payload)
        return deepcopy(outcome)

    handler = make_handler(tmp_path, model)
    snapshot = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
    state = {
        "confirmed_config_version": snapshot["config_version"],
        "confirmed_config": deepcopy(snapshot["config"]),
        "tasks": [{"task_id": "t1", "status": "pending"}],
    }
    write_json(handler.state_path, state)

    def fail(current, **dependencies):
        if partial_commit:
            current = deepcopy(current)
            current["tasks"][0]["status"] = "completed"
            write_json(handler.state_path, current)
        raise ValueError("synthetic invalid return")

    monkeypatch.setattr(recovery, "recover_saved_results", fail)
    handler._run = lambda *args, **kwargs: pytest.fail("Science cannot run after recovery failure")
    handler.config_revision_factory = lambda *args: pytest.fail("No draft may be created")
    return handler, requests, state


@pytest.mark.parametrize("kind", ["answer", "inspect_configuration"])
def test_recovery_failure_preserves_semantic_readonly_dialogue(tmp_path, monkeypatch, kind):
    handler, requests, state = recovery_handler(
        tmp_path, monkeypatch, {"kind": kind, "answer": "依据已保存事实答复"}
    )
    reply = handler([{"role": "user", "content": "参数给我看看"}], conversation_id="c")
    assert "synthetic invalid return" in reply
    assert "回收" in reply
    if kind == "inspect_configuration":
        assert "本次仅查看" in reply or "配置" in reply
    else:
        assert "依据已保存事实答复" in reply
    assert requests and requests[0]["allowed_tools"] == []
    assert "失败" in requests[0]["decision_context"]["saved_facts_notice"]
    assert read_json(handler.state_path, {}) == state
    assert handler.config_delegate is None


def test_recovery_failure_uses_committed_facts_after_partial_save(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue import decision

    handler, requests, state = recovery_handler(
        tmp_path,
        monkeypatch,
        {"kind": "answer", "answer": "只读解释"},
        partial_commit=True,
    )
    observed = []
    original = decision.decide_from_saved_facts

    def capture(handler, facts, *args, **kwargs):
        observed.append(deepcopy(facts))
        return original(handler, facts, *args, **kwargs)

    monkeypatch.setattr(decision, "decide_from_saved_facts", capture)
    reply = handler([{"role": "user", "content": "解释上次失败原因"}], conversation_id="c")
    assert "synthetic invalid return" in reply
    assert observed[0]["tasks"][0]["status"] == "completed"
    assert read_json(handler.state_path, {})["tasks"][0]["status"] == "completed"


@pytest.mark.parametrize(
    "outcome",
    [
        {"kind": "advance"},
        {"kind": "configure", "answer": "修改配置"},
        {"kind": "review", "plan_id": "p", "decision": "approve"},
        {"kind": "phase_diagram", "operation": "plot"},
        {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "核对"},
    ],
)
def test_recovery_failure_blocks_model_controls_and_science(tmp_path, monkeypatch, outcome):
    handler, requests, state = recovery_handler(tmp_path, monkeypatch, outcome)
    reply = handler([{"role": "user", "content": "继续"}], conversation_id="c")
    assert "synthetic invalid return" in reply
    assert read_json(handler.state_path, {}) == state
    assert handler.config_delegate is None


def test_independent_new_run_skips_failed_old_recovery(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import recover_saved_results as recovery

    handler, requests, state = recovery_handler(
        tmp_path, monkeypatch, {"kind": "answer", "answer": "unused"}
    )
    handler.history_decision = None
    handler.history_prompt = True
    new_path = tmp_path / "independent" / "state.json"
    calls = []
    handler.new_run_factory = lambda: calls.append("new") or {"state_path": new_path}
    monkeypatch.setattr(
        recovery,
        "recover_saved_results",
        lambda *args, **kwargs: pytest.fail("New run cannot recover or modify the old project"),
    )
    old_path = handler.state_path
    reply = handler([{"role": "user", "content": "新建"}], conversation_id="c")
    assert "已新建独立运行" in reply
    assert handler.state_path == new_path
    assert calls == ["new"] and not requests
    assert read_json(old_path, {}) == state
