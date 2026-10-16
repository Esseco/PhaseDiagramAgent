from copy import deepcopy

import pytest

from phase_agent.tools.local.prepare_dft_upload_batches import _prepare_batches
from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner


@pytest.mark.parametrize("version", ["original", "deferred", "current"])
def test_batch_loop_accepts_loaded_older_prepare_signatures(version):
    calls = []

    def result(state):
        calls.append(True)
        current = deepcopy(state)
        if len(calls) == 1:
            current["tasks"].append("prepared")
            return {"status": "prepared", "state": current, "batch": {"batch_id": "b"}}
        return {"status": "no_tasks", "state": current}

    class Runner:
        if version == "original":

            def prepare(self, state):
                return result(state)
        elif version == "deferred":

            def prepare(self, state, *, _defer_index=False, _defer_archive=False):
                assert _defer_index and _defer_archive
                return result(state)
        else:

            def prepare(self, state, *, _owned=False, _defer_index=False, _defer_archive=False):
                assert _owned and _defer_index and _defer_archive
                return result(state)

    source = {"tasks": []}
    state, batches = _prepare_batches(Runner(), source)
    assert state["tasks"] == ["prepared"]
    assert source == {"tasks": []}
    assert batches == [{"batch_id": "b"}]
    assert len(calls) == 2


def test_internal_typeerror_is_propagated_without_repeating_prepare():
    calls = []

    class Runner:
        def prepare(self, state, **kwargs):
            calls.append(True)
            raise TypeError("internal preparation failure")

    with pytest.raises(TypeError, match="internal preparation failure"):
        _prepare_batches(Runner(), {"tasks": []})
    assert calls == [True]


def test_current_manual_runner_accepts_loaded_original_parent(tmp_path, monkeypatch):
    calls = []

    def original_prepare(self, state):
        calls.append(True)
        return {"status": "no_tasks", "state": deepcopy(state), "batch": None}

    monkeypatch.setattr(RemoteBatchRunner, "prepare", original_prepare)
    runner = ManualUploadBatchRunner(tmp_path, worker_command=[])
    source = {"tasks": []}
    result = runner.prepare(source, _owned=True, _defer_index=True, _defer_archive=True)
    assert result["state"] == source
    assert result["state"] is not source
    assert calls == [True]
