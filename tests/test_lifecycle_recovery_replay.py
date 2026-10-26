from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib

from phase_agent.graphs.project.runner import run_search_workflow_graph


class _NoopLock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoopSaver:
    @classmethod
    def from_conn_string(cls, *args, **kwargs):
        return cls()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _CheckpointGraph:
    def __init__(self):
        self.checkpointer = None
        self.threads = []

    def get_state(self, config, subgraphs=False):
        thread = config["configurable"]["thread_id"]
        self.threads.append(thread)
        if ":recovery:" in thread:
            return SimpleNamespace(next=(), values={}, tasks=())
        return SimpleNamespace(
            next=("execute",),
            values={"request_id": "old-request"},
            tasks=(SimpleNamespace(interrupts=()),),
        )

    def invoke(self, value, config, context=None):
        context.response = {
            "status": "execution_reconciliation_required",
            "submitted": False,
        }


def test_unsafe_stale_checkpoint_uses_read_only_recovery_lifecycle(tmp_path):
    graph = _CheckpointGraph()
    stages = {
        "initialize": lambda: {"_workflow_prepared": True, "snapshot": {}},
        "collect": lambda frame: frame,
        "analyze": lambda frame: frame,
        "wait": lambda frame: frame,
        "assess": lambda frame: frame,
        "act": lambda frame: frame,
        "finalize": lambda frame: frame,
    }

    with patch("filelock.FileLock", _NoopLock), patch(
        "phase_agent.persistence.compact_checkpoints.CompactSqliteSaver.from_conn_string",
        _NoopSaver.from_conn_string,
    ), patch("phase_agent.tools.state.workflow_control.current_revision", lambda path: None):
        result = run_search_workflow_graph(
            workflow_graph=graph,
            checkpoint_path=Path(tmp_path) / "lifecycle.sqlite",
            business_state_path=Path(tmp_path) / "state.json",
            invocation_id="new-request",
            project_thread="project-scientific-lifecycle-v2",
            **stages,
        )

    assert result["status"] == "execution_reconciliation_required"
    assert graph.threads == [
        "project-scientific-lifecycle-v2",
        "project-scientific-lifecycle-v2:recovery:"
        + hashlib.sha256(b"new-request").hexdigest()[:16],
    ]


def test_clean_business_state_fences_stale_checkpoint_and_starts_fresh_request(tmp_path):
    graph = _CheckpointGraph()
    state_path = Path(tmp_path) / "state.json"
    state_path.write_text(
        '{"execution_recovery_report":{"status":"clear","unsettled":[]},'
        '"pending_execution_policies":{},"pending_execution_recoveries":{},'
        '"budget_reservations":{},"tasks":[]}',
        encoding="utf-8",
    )
    stages = {
        "initialize": lambda: {"_workflow_prepared": True, "snapshot": {}},
        "collect": lambda frame: frame,
        "analyze": lambda frame: frame,
        "wait": lambda frame: frame,
        "assess": lambda frame: frame,
        "act": lambda frame: frame,
        "finalize": lambda frame: frame,
    }

    with patch("filelock.FileLock", _NoopLock), patch(
        "phase_agent.persistence.compact_checkpoints.CompactSqliteSaver.from_conn_string",
        _NoopSaver.from_conn_string,
    ), patch("phase_agent.tools.state.workflow_control.current_revision", lambda path: None):
        result = run_search_workflow_graph(
            workflow_graph=graph,
            checkpoint_path=Path(tmp_path) / "lifecycle.sqlite",
            business_state_path=state_path,
            invocation_id="new-request",
            project_thread="project-scientific-lifecycle-v2",
            **stages,
        )

    assert result["status"] == "execution_reconciliation_required"
    assert graph.threads == ["project-scientific-lifecycle-v2"]


def test_superseded_reservations_do_not_fence_clean_current_request(tmp_path):
    from phase_agent.graphs.project.runner import _business_state_is_clean_for_stale_checkpoint

    state_path = Path(tmp_path) / "state.json"
    state_path.write_text(
        '{"confirmed_config_version":"config-new",'
        '"execution_recovery_report":{"status":"clear","unsettled":[]},'
        '"pending_execution_policies":{},"pending_execution_recoveries":{},'
        '"budget_reservations":{"old":{"config_version":"config-old",'
        '"status":"reserved"}},"tasks":[]}',
        encoding="utf-8",
    )

    assert _business_state_is_clean_for_stale_checkpoint(state_path)
