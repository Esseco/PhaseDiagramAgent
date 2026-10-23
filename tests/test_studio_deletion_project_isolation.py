"""Delete synthetic Studio history while retaining actual project stores and approvals."""

import asyncio
from copy import deepcopy
import hashlib

from blockbuster import blockbuster_ctx
import pytest

from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
from phase_agent.persistence.memory.conversation import memory_path, recent_turns, remember_turn
from phase_agent.persistence.memory.decision_memory import update_long_term_advice
from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.state.execution_receipts import begin_execution, recovery_report
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.workflows.run_event_loop import run_event_loop
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from tests.test_studio_storage import checkpoint
from tests.test_studio_thread_deletion import native  # noqa: F401


def hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def delete_old(native, parallel=False):
    async def erase():
        with blockbuster_ctx(scanned_modules=["phase_agent", "langgraph_runtime_inmem"]):
            if parallel:
                results = await asyncio.gather(
                    *(native.ops.Threads.delete(native.conn, tid) for tid in native.ids[:2])
                )
            else:
                results = [
                    await native.ops.Threads.delete(native.conn, tid) for tid in native.ids[:2]
                ]
            return [await anext(result) for result in results]

    assert set(asyncio.run(erase())) == set(native.ids[:2])


def setup_policy():
    from tests.test_execution_policy import ExecutionPolicyTest

    setup = ExecutionPolicyTest()
    setup.setUp()
    return setup


@pytest.mark.parametrize("parallel", [False, True])
def test_old_batch_preserves_active_task_approval_receipts_and_project_memory(
    native, tmp_path, parallel
):
    n = native
    path = tmp_path / "project/workflow_state/state.json"
    setup = setup_policy()
    base = update_long_term_advice(
        {
            "tasks": [
                {
                    "task_id": "active-dft",
                    "stage": "dft_single_point",
                    "status": "running",
                    "batch_id": "batch-current",
                    "model_version": "m1",
                }
            ]
        },
        ["Use broad DFT coverage when D and H are high"],
        source="explicit_user",
    )
    base["decision_memory"]["records"] = [
        {
            "knowledge_id": "user-rule",
            "scope": "user",
            "status": "active",
            "approved_by_user": True,
            "maturity": "heuristic",
            "statement": "Keep endpoint protection",
        }
    ]
    invocation = "old-chat:" + str(n.ids[0])
    first = run_tool_step(
        base,
        setup.session,
        registry=setup.registry,
        agent_client=setup._agent,
        execution_mode="interactive",
        invocation_id=invocation,
        context={"state_path": str(path)},
    )
    assert first["status"] == "awaiting_approval"
    write_json(path, first["state"])
    for tid in n.ids:
        remember_turn(path, str(tid), "saved question", "saved answer " + str(tid))
    identity = {
        "invocation_id": "running-task-receipt",
        "config_version": "cfg",
        "action_hash": "unchanged",
        "tool": "run_calculation_stage",
    }
    assert begin_execution(path, identity)["allowed"]
    lifecycle = path.parent / "langgraph_lifecycle.sqlite"
    cfg = {"configurable": {"thread_id": "project-scientific-lifecycle-v2", "checkpoint_ns": ""}}
    with CompactSqliteSaver.from_conn_string(str(lifecycle)) as saver:
        saved = saver.put(
            cfg, checkpoint(1, {"loaded_state": first["state"]}), {}, {"business_frame": 1}
        )
        frame = saver.get_tuple(saved)
    n.conn.store["runs"][-1]["status"] = "running"
    n.conn.store["threads"][-1]["status"] = "busy"
    current = n.memory.get_tuple(n.saved[n.ids[-1]])
    current_run = deepcopy(n.conn.store["runs"][-1])
    handler = RunWorkflowChatHandler({"state_path": str(path), "config_session": setup.session})
    control = LocalAgentControl(handler)
    before_views = (control.tasks(), control.pending(), control.memory())
    before_files = hashes(path.parent)
    before_knowledge = retrieve_relevant_knowledge(read_json(path))
    pending = deepcopy(first["state"]["pending_execution_policies"])

    delete_old(n, parallel)

    assert hashes(path.parent) == before_files
    reloaded = read_json(path)
    assert reloaded["tasks"][0]["status"] == "running"
    assert reloaded["pending_execution_policies"] == pending
    fresh = LocalAgentControl(
        RunWorkflowChatHandler({"state_path": str(path), "config_session": setup.session})
    )
    assert (fresh.tasks(), fresh.pending(), fresh.memory()) == before_views
    assert n.memory.get_tuple(n.saved[n.ids[-1]]) == current
    assert n.conn.store["runs"] == [current_run]
    assert [row["thread_id"] for row in n.conn.store["threads"]] == n.ids[2:]
    assert all(str(tid) not in n.memory.storage for tid in n.ids[:2])
    assert recent_turns(path, str(n.ids[-1]))[-1]["assistant"].endswith(str(n.ids[-1]))
    assert all(memory_path(path, str(tid)).is_file() for tid in n.ids)
    assert retrieve_relevant_knowledge(reloaded) == before_knowledge
    assert any(row["knowledge_id"] == "user-rule" for row in before_knowledge)
    assert recovery_report(path, reloaded)["status"] == "reconciliation_required"
    assert not begin_execution(path, identity)["allowed"]
    with CompactSqliteSaver.from_conn_string(str(lifecycle)) as saver:
        assert saver.get_tuple(saved) == frame
    assert setup.calls == []


def test_deleted_origin_chat_can_review_original_project_plan_once(native, tmp_path):
    n = native
    path = tmp_path / "project/workflow_state/state.json"
    setup = setup_policy()
    invocation = "old-chat:" + str(n.ids[0])
    first = run_event_loop(
        {},
        setup.session,
        registry=setup.registry,
        agent_client=setup._agent,
        execution_mode="interactive",
        invocation_id=invocation,
        state_path=path,
    )
    assert first["status"] == "awaiting_approval" and setup.calls == []
    before = hashes(path.parent)
    delete_old(n)
    assert hashes(path.parent) == before

    def no_new_model(*args, **kwargs):
        pytest.fail("Restoring an original approval must not request a new model decision")

    def workflow(**kw):
        return run_event_loop(
            kw["state_path"],
            kw["config_session"],
            registry=setup.registry,
            agent_client=no_new_model,
            execution_mode="interactive",
            invocation_id=kw["invocation_id"],
            state_path=kw["state_path"],
            human_feedback=kw.get("human_feedback"),
        )

    handler = RunWorkflowChatHandler(
        {"state_path": str(path), "config_session": setup.session}, workflow=workflow
    )
    handler.conversation_id = str(n.ids[-1])
    row = LocalAgentControl(handler).pending()["pending"][0]
    assert row["plan_id"] == invocation
    with pytest.raises(ValueError, match="proposal_hash_mismatch"):
        handler.review_pending(
            invocation,
            "approve",
            expected_state_version=row["state_version"],
            expected_proposal_hash="wrong",
        )
    assert setup.calls == []
    result = handler.review_pending(
        invocation,
        "approve",
        expected_state_version=row["state_version"],
        expected_proposal_hash=row["proposal_hash"],
    )
    assert result["status"] == "completed" and len(setup.calls) == 1
    again = handler.review_pending(
        invocation,
        "approve",
        expected_state_version=row["state_version"],
        expected_proposal_hash=row["proposal_hash"],
    )
    assert again["status"] == "already_processed" and len(setup.calls) == 1


@pytest.mark.parametrize("status", ["running", "pending"])
def test_native_status_filter_finds_active_run_behind_hundred_successes(
    native, monkeypatch, status
):
    from unittest.mock import AsyncMock
    from uuid import UUID

    n = native
    monkeypatch.setattr(n.ops.Runs, "handle_event", AsyncMock(return_value=None))
    old = {
        "thread_id": n.ids[0],
        "run_id": UUID(int=999),
        "status": status,
        "metadata": {},
        "created_at": 0,
    }
    n.conn.store["runs"] = [old] + [
        {**old, "run_id": UUID(int=1000 + i), "status": "success", "created_at": i + 1}
        for i in range(100)
    ]

    async def query():
        recent = await n.ops.Runs.search(n.conn, n.ids[0], limit=100)
        filtered = await n.ops.Runs.search(n.conn, n.ids[0], limit=1, status=status)
        return [row async for row in recent], [row async for row in filtered]

    recent, filtered = asyncio.run(query())
    assert len(recent) == 100 and all(row["status"] == "success" for row in recent)
    assert filtered == [old]
