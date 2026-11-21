import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from filelock import Timeout

from phase_agent.runtime.codex.offline import CodexOffline


def test_recovery_obeys_project_lifecycle_lock_across_processes(tmp_path, monkeypatch):
    first = tmp_path / "first" / "state.json"
    second = tmp_path / "second" / "state.json"
    first.parent.mkdir()
    second.parent.mkdir()
    lock = str(first.parent / "langgraph_lifecycle.sqlite") + ".lock"
    code = (
        'import sys,time; from filelock import FileLock; '
        'lock=FileLock(sys.argv[1]); lock.acquire(); print("ready",flush=True); time.sleep(60)'
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, lock], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        offline = CodexOffline.__new__(CodexOffline)
        offline.handler = SimpleNamespace(
            lock=threading.RLock(), state_path=first, workflow_kwargs={}
        )
        offline.control = SimpleNamespace(_codex_context_cache="old")
        with pytest.raises(Timeout):
            offline.recover()

        monkeypatch.setattr(
            "phase_agent.tools.step_runner.file_protocol.read_json", lambda *args: {}
        )
        monkeypatch.setattr(
            "phase_agent.tools.workflows.recover_saved_results.recover_saved_results",
            lambda state, **kwargs: state,
        )
        monkeypatch.setattr(
            "phase_agent.tools.local.recover_remote_training.recover_remote_training",
            lambda state: (state, []),
        )
        offline.handler.state_path = second
        assert offline.recover()["status"] == "observed"
        assert offline.control._codex_context_cache is None
    finally:
        holder.terminate()
        holder.wait(timeout=10)
