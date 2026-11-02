"""Small real-SQLite lifecycle regressions; external scientific steps are synthetic."""

from copy import copy
from contextlib import closing
import hashlib
import sqlite3

import pytest
from langchain_core.messages import HumanMessage

from phase_agent.graphs.project.runner import run_search_workflow_graph
from phase_agent.graphs.scientific_graph import scientific_graph
from phase_agent.graphs.invocation_context import new_invocation
from phase_agent.graphs.studio_chat_graph import build_studio_graph
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
from phase_agent.persistence.lifecycle_routes import (
    current_lifecycle_thread,
    fresh_lifecycle_thread,
)
from phase_agent.runtime.studio_recovery import can_resume_message
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_skill_recovery_boundaries import lifecycle


def poison(options, *, thread="project", recovery_cache=False):
    with CompactSqliteSaver.from_conn_string(str(options["checkpoint_path"])) as saver:
        graph = copy(scientific_graph())
        graph.checkpointer = saver
        config = {"configurable": {"thread_id": thread}}
        graph.update_state(config, {"request_id": "old", "business_frame": {}}, as_node="analyze")
        assert graph.get_state(config).next == ("react",)
        if recovery_cache:
            config["configurable"]["thread_id"] += (
                ":recovery:" + hashlib.sha256(b"new").hexdigest()[:16]
            )
            graph.update_state(
                config,
                {
                    "request_id": "old",
                    "completed_response": {"status": "OLD_SUCCESS", "submitted": True},
                },
                as_node="finalize",
            )


def case(tmp_path, failed="never", status="done"):
    options, calls, failure = lifecycle(tmp_path, failed, status)
    options["checkpoint_path"] = tmp_path / "langgraph_lifecycle.sqlite"
    options["business_state_path"] = tmp_path / "state.json"
    write_json(
        options["business_state_path"],
        {
            "execution_recovery_report": {"status": "clear", "unsettled": []},
            "tasks": [],
        },
    )
    return options, calls, failure


def test_fresh_completion_retry_never_recollects(tmp_path):
    options, calls, _ = case(tmp_path)
    poison(options)
    result = run_search_workflow_graph(invocation_id="new", **options)
    previous = list(calls)
    assert run_search_workflow_graph(invocation_id="new", **options) == result
    assert calls == previous
    with closing(sqlite3.connect(options["checkpoint_path"])) as connection, connection:
        assert current_lifecycle_thread(connection, "project") == fresh_lifecycle_thread(
            "project", "new"
        )


@pytest.mark.parametrize("failed", ["observe", "assess"])
@pytest.mark.parametrize("same_id", [True, False])
def test_fresh_failed_read_resumes_after_reopening_database(tmp_path, failed, same_id):
    options, calls, failure = case(tmp_path, failed)
    poison(options)
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run_search_workflow_graph(invocation_id="new", **options)
    failure[0] = False
    result = run_search_workflow_graph(invocation_id="new" if same_id else "retry", **options)
    assert result == {"status": "done"}
    assert calls.count("collect") == 1


def test_old_recovery_success_cannot_hide_current_block(tmp_path):
    options, calls, _ = case(tmp_path)
    poison(options, recovery_cache=True)
    state = read_json(options["business_state_path"], {})
    state["tasks"] = [{"task_id": "real-inflight", "status": "running"}]
    write_json(options["business_state_path"], state)
    result = run_search_workflow_graph(invocation_id="new", **options)
    assert result["status"] == "execution_reconciliation_required"
    assert result["submitted"] is False and calls == []


def test_fresh_approval_then_pending_tasks_then_return_stays_on_current_thread(tmp_path):
    options, calls, _ = case(tmp_path, status="awaiting_approval")
    poison(options)
    path = options["business_state_path"]
    feedback = [False]
    returned = [False]
    executions = []

    def act(frame, **kwargs):
        state = read_json(path, {})
        if not feedback[0]:
            state["pending_execution_policies"] = {"plan": {"agent_proposal": {}}}
        else:
            executions.append("inputs")
            state["pending_execution_policies"] = {}
            state["tasks"] = [{"task_id": "new-task", "status": "pending"}]
        write_json(path, state)
        return frame

    def wait(frame):
        if read_json(path, {}).get("tasks") and not returned[0]:
            return {"status": "awaiting_results"}
        return frame

    options.update(
        act=act,
        wait=wait,
        finalize=lambda f: {
            "status": "done"
            if returned[0]
            else "awaiting_results"
            if feedback[0]
            else "awaiting_approval"
        },
    )
    assert (
        run_search_workflow_graph(invocation_id="proposal", **options)["status"]
        == "awaiting_approval"
    )
    feedback[0] = True
    assert (
        run_search_workflow_graph(
            invocation_id="plan", human_feedback={"decision": "approve"}, **options
        )["status"]
        == "awaiting_results"
    )
    assert executions == ["inputs"]
    assert (
        run_search_workflow_graph(invocation_id="waiting", **options)["status"]
        == "awaiting_results"
    )
    assert executions == ["inputs"]
    returned[0] = True
    # Substitute the next scientific decision after observation, not another input execution.
    options["act"] = lambda frame, **kw: frame
    assert run_search_workflow_graph(invocation_id="returned", **options)["status"] == "done"
    assert executions == ["inputs"]


@pytest.mark.parametrize("control", [False, True])
def test_studio_retry_finds_fresh_read_checkpoint_and_latest_reply(tmp_path, monkeypatch, control):
    options, calls, failure = case(tmp_path, "observe")
    options["project_thread"] = "project-scientific-lifecycle-v2"
    if control:
        state_path = options["business_state_path"]
        state = read_json(state_path, {})
        state["workflow_control_revision"] = "r1"
        write_json(state_path, state)
        write_json(
            str(state_path) + ".control.json",
            {"state_path": str(state_path.resolve()), "revision": "r1"},
        )
    base = options["project_thread"] + (":control:r1" if control else "")
    poison(options, thread=base)
    config_path = tmp_path / "agent_runtime.json"
    write_json(config_path, {"state_path": str(options["business_state_path"])})
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config_path))

    def sender(text, thread):
        return run_search_workflow_graph(invocation_id=new_invocation(), **options)["status"]

    graph = build_studio_graph(sender, receipt_path=tmp_path / "gateway.json")
    request = {"messages": [HumanMessage(content="继续", id="m")]}
    config = {"configurable": {"thread_id": "studio"}}
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        graph.invoke(request, config)
    assert can_resume_message(config_path, "studio:m")
    before = options["checkpoint_path"].read_bytes()
    assert can_resume_message(config_path, "studio:m")
    assert options["checkpoint_path"].read_bytes() == before
    failure[0] = False
    result = graph.invoke(request, config)
    assert result["messages"][-1].content == "done"
    assert calls.count("collect") == 1
    previous = list(calls)
    assert graph.invoke(request, config)["messages"][-1].content == "done"
    assert calls == previous


def test_pre_route_legacy_fresh_is_adopted_for_next_message(tmp_path):
    options, calls, _ = case(tmp_path, status="awaiting_results")
    poison(options)
    run_search_workflow_graph(invocation_id="new", **options)
    with closing(sqlite3.connect(options["checkpoint_path"])) as connection, connection:
        connection.execute("DROP TABLE lifecycle_routes")
    state = read_json(options["business_state_path"], {})
    state["tasks"] = [{"task_id": "active", "status": "pending"}]
    write_json(options["business_state_path"], state)
    assert (
        run_search_workflow_graph(invocation_id="continue", **options)["status"]
        == "awaiting_results"
    )


def test_corrupt_route_does_not_replay_or_fallback(tmp_path):
    options, calls, _ = case(tmp_path)
    poison(options)
    with closing(sqlite3.connect(options["checkpoint_path"])) as connection, connection:
        connection.execute(
            "CREATE TABLE lifecycle_routes(base_thread TEXT PRIMARY KEY, active_thread TEXT)"
        )
        connection.execute("INSERT INTO lifecycle_routes VALUES('project','another-project')")
    with pytest.raises(ValueError, match="路由"):
        run_search_workflow_graph(invocation_id="new", **options)
    assert calls == []


def test_same_approved_fresh_request_resumes_failed_read(tmp_path):
    options, calls, failure = case(tmp_path, "observe")
    poison(options)
    state = read_json(options["business_state_path"], {})
    state["pending_execution_policies"] = {"new": {"agent_proposal": {}}}
    write_json(options["business_state_path"], state)
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run_search_workflow_graph(
            invocation_id="new", human_feedback={"decision": "approve"}, **options
        )
    failure[0] = False
    assert run_search_workflow_graph(
        invocation_id="new", human_feedback={"decision": "approve"}, **options
    ) == {"status": "done"}
    assert calls.count("collect") == 1 and calls.count("act") == 1


@pytest.mark.parametrize("existing_route", [False, True])
def test_start_checkpoint_survives_crash_before_route_commit(tmp_path, monkeypatch, existing_route):
    from phase_agent.persistence import lifecycle_routes

    options, calls, _ = case(tmp_path)
    poison(options)
    if existing_route:
        with closing(sqlite3.connect(options["checkpoint_path"])) as connection, connection:
            lifecycle_routes.activate_lifecycle_thread(connection, "project", "project")
    actual_activate = lifecycle_routes.activate_lifecycle_thread

    def crash(*args):
        raise RuntimeError("synthetic route commit interruption")

    monkeypatch.setattr(lifecycle_routes, "activate_lifecycle_thread", crash)
    with pytest.raises(RuntimeError, match="route commit interruption"):
        run_search_workflow_graph(invocation_id="new", **options)
    assert calls == []
    monkeypatch.setattr(lifecycle_routes, "activate_lifecycle_thread", actual_activate)
    assert run_search_workflow_graph(invocation_id="new", **options) == {"status": "done"}
    assert calls.count("collect") == 1
