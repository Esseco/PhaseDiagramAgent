import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.tools.relax_mc_batch.run import SlurmIO
from phase_agent.tools.relax_mc_batch.files import read
from tests.test_relax_mc_phase_dependencies import setup, run_batch


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("f")


@pytest.mark.parametrize(
    "state,active", [("COMPLETED", set()), ("FAILED", set()), ("RUNNING", {"3741924"})]
)
def test_accounting_confirms_state_when_squeue_fails(monkeypatch, state, active):
    import subprocess

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "squeue":
            return SimpleNamespace(
                returncode=1, stdout="", stderr="slurm_load_jobs error: Invalid job id specified"
            )
        return SimpleNamespace(returncode=0, stdout=f"3741924|{state}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert SlurmIO().active(["3741924"]) == active
    assert [c[0][0] for c in calls] == ["squeue", "sacct"]
    assert all(c[1]["timeout"] <= 30 for c in calls)


def test_scheduler_unknown_is_diagnostic_and_not_completed(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="controller unavailable"
        ),
    )
    from phase_agent.tools.relax_mc_batch.run import SchedulerQueryError

    with pytest.raises(SchedulerQueryError) as caught:
        SlurmIO().active(["3741924"])
    assert caught.value.diagnostics[0]["stderr"] == "controller unavailable"
    assert caught.value.diagnostics[0]["returncode"] == 1


def test_missing_accounting_row_is_unknown_not_finished(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1 if command[0] == "squeue" else 0, stdout="", stderr="not found"
        ),
    )
    from phase_agent.tools.relax_mc_batch.run import SchedulerQueryError

    with pytest.raises(SchedulerQueryError):
        SlurmIO().active(["3741924"])


def test_array_member_is_not_mistaken_for_complete_parent(monkeypatch):
    import subprocess
    from phase_agent.tools.relax_mc_batch.run import SchedulerQueryError

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1 if command[0] == "squeue" else 0,
            stdout="3741924_0|COMPLETED\n3741924_1|RUNNING\n" if command[0] == "sacct" else "",
            stderr="invalid job",
        ),
    )
    with pytest.raises(SchedulerQueryError):
        SlurmIO().active(["3741924"])


def test_scheduler_timeouts_keep_partial_diagnostics(monkeypatch):
    import subprocess
    from phase_agent.tools.relax_mc_batch.run import SchedulerQueryError

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 20, stderr="connection timed out")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(SchedulerQueryError) as error:
        SlurmIO().active(["3741924"])
    assert len(error.value.diagnostics) == 2
    assert all(r["stderr"] == "connection timed out" for r in error.value.diagnostics)


def test_no_verified_results_does_not_rebuild_hull(tmp_path, monkeypatch):
    from phase_agent.tools.relax_mc_batch import dependencies
    from phase_agent.tools.relax_mc_batch.run import SchedulerQueryError
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner

    state = {
        "tasks": [{"task_id": "t", "status": "pending"}],
        "slurm_batches": [{"batch_id": "b", "job_id": "3741924", "task_ids": ["t"]}],
        "relax_mc_remote": {"phase": 0, "phase_controllers": {"0": {"batch_ids": ["b"]}}},
    }
    monkeypatch.setattr(ManualUploadBatchRunner, "collect_results", lambda *args: [])
    feedback = Mock(side_effect=AssertionError("No empty hull update"))
    monkeypatch.setattr(dependencies, "feedback", feedback)
    io = SimpleNamespace(active=Mock(side_effect=SchedulerQueryError([])))
    current, results = dependencies.collect_phase(
        tmp_path, state, None, {"config": {"supercomputer": {"worker": {"command": []}}}}, io
    )
    assert results == [] and current["tasks"][0]["status"] == "pending"
    feedback.assert_not_called()


def test_unknown_scheduler_keeps_verified_results_and_pending_missing(tmp_path, monkeypatch):
    from phase_agent.tools.relax_mc_batch.run import SchedulerQueryError

    root, io, calls = setup(tmp_path, monkeypatch)
    run_batch(root, submit=True, io=io)
    before = read(root / "workflow_state.json")
    submitted = len(io.rows)
    # A partial callback validates the real fake-worker receipts, then exposes a
    # subset, representing a phase where other worker files have not arrived.
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner

    original = ManualUploadBatchRunner.collect_results

    def collect(self, state):
        return original(self, state)[:2]

    monkeypatch.setattr(ManualUploadBatchRunner, "collect_results", collect)

    def unknown(ids):
        raise SchedulerQueryError(
            [{"command": ["squeue"], "returncode": 1, "stderr": "controller unavailable"}]
        )

    io.active = unknown
    outcome = run_batch(root, submit=True, io=io, phase=0)
    assert outcome["status"] == "partial", outcome
    assert len(io.rows) == submitted
    current = read(root / "workflow_state.json")
    tasks = current["tasks"]
    assert sum(t["status"] == "completed" for t in tasks) == 2
    assert not any(t["status"] == "failed" for t in tasks)
    assert sum(t["status"] == "pending" for t in tasks) == len(before["tasks"]) - 2
    assert current["relax_mc_remote"]["phase"] == 0
    assert (
        current["relax_mc_remote"]["scheduler_query_error"]["diagnostics"][0]["stderr"]
        == "controller unavailable"
    )
    assert not any(b.get("relax_mc_wave_done") for b in current["slurm_batches"])
    assert (root / "results.zip").is_file()
    assert run_batch(root, submit=True, io=io)["submitted"] is False


def test_empty_studio_resume_is_reported_without_dispatch():
    from phase_agent.runtime.studio_payloads import guarded_studio_stream

    run = {
        "kwargs": {
            "input": None,
            "command": None,
            "stream_mode": ["messages", "debug"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }
    before = deepcopy(run)

    async def empty(*args, **kwargs):
        yield "debug", {"type": "checkpoint"}

    async def check():
        with pytest.raises(ValueError, match="没有可恢复节点"):
            async for _ in guarded_studio_stream(empty, run, 1, None):
                pass

    asyncio.run(check())
    assert run == before


@pytest.mark.parametrize(
    "update", [{"agent": {"reply": "ok"}}, {"__interrupt__": [{"value": "approval"}]}]
)
def test_legitimate_studio_resume_and_interrupt_are_preserved(update):
    from phase_agent.runtime.studio_payloads import guarded_studio_stream

    run = {
        "kwargs": {
            "input": None,
            "command": None,
            "stream_mode": ["messages"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }

    async def original(prepared, *args, **kwargs):
        assert "updates" in prepared["kwargs"]["stream_mode"]
        yield "updates", update
        yield "messages", {"reply": "ok"}

    async def check():
        return [row async for row in guarded_studio_stream(original, run, 1, None)]

    assert asyncio.run(check()) == [("messages", {"reply": "ok"})]


def test_guard_with_real_langgraph_checkpoint_resume():
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from phase_agent.runtime.studio_payloads import guarded_studio_stream

    class State(TypedDict):
        value: int

    invoked = Mock(return_value={"value": 2})
    builder = StateGraph(State)
    builder.add_node("work", invoked)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["work"])
    config = {"configurable": {"thread_id": "offline-resume"}}
    graph.invoke({"value": 1}, config)
    invoked.assert_not_called()
    run = {
        "kwargs": {
            "input": None,
            "command": None,
            "stream_mode": ["values"],
            "config": {"configurable": {"graph_id": "phase_chat"}},
        }
    }

    async def original(prepared, *args, **kwargs):
        async for mode, event in graph.astream(
            None, config, stream_mode=prepared["kwargs"]["stream_mode"]
        ):
            yield mode, event

    async def check():
        first = [row async for row in guarded_studio_stream(original, run)]
        assert first[-1][1]["value"] == 2
        with pytest.raises(ValueError, match="没有可恢复节点"):
            async for _ in guarded_studio_stream(original, run):
                pass

    asyncio.run(check())
    assert invoked.call_count == 1


def test_imported_partial_batch_does_not_recommend_whole_batch_resubmission():
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    state = {
        "tasks": [
            {
                "task_id": "mc",
                "stage": "deep_search",
                "status": "pending",
                "relax_mc_bundle_id": "b",
            }
        ],
        "relax_mc_batches": {
            "b": {
                "directory": "original_batch",
                "import_checksum": "verified",
                "result_summary": {"status": "failed", "reason": "squeue query failed"},
            }
        },
    }
    before = deepcopy(state)
    text = "\n".join(waiting_file_lines(state))
    assert "仍缺 1 项结果的核验凭据" in text
    assert "不能证明计算失败" in text
    assert "不要重新上传输入或提交整批" in text
    assert "提交：" not in text and "sbatch" not in text
    assert state == before
