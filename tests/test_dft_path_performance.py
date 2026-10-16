import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from phase_agent.tools.remote.transfer_bundle import MANIFEST, create_bundle


def test_bundle_reads_payload_once_and_hashes_archived_bytes(tmp_path, monkeypatch):
    source = tmp_path / "data.txt"
    source.write_bytes(b"original")
    original = Path.open
    reads = []

    def opening(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            reads.append(path)
            source.write_bytes(b"changed after stat")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    archive = create_bundle(tmp_path, [source], tmp_path / "inputs.zip", kind="inputs")
    assert reads == [source]
    with zipfile.ZipFile(archive) as zipped:
        row = json.loads(zipped.read(MANIFEST))["files"][0]
        actual = zipped.read("data.txt")
        assert actual == b"changed after stat"
        assert row["size"] == len(actual)
        assert row["sha256"] == hashlib.sha256(actual).hexdigest()
        assert zipped.testzip() is None


def test_bundle_failure_preserves_destination_and_removes_partial(tmp_path, monkeypatch):
    source = tmp_path / "data.txt"
    source.write_bytes(b"input")
    target = tmp_path / "inputs.zip"
    target.write_bytes(b"previous archive")
    original = Path.open

    class Broken(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("input read failed")
            return super().read(size)

    def opening(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            return Broken(b"first chunk")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(OSError, match="input read failed"):
        create_bundle(tmp_path, [source], target, kind="inputs")
    assert target.read_bytes() == b"previous archive"
    assert not target.with_name("inputs.zip.part").exists()


def test_owned_action_state_still_has_one_outer_isolation_boundary():
    from phase_agent.tools.workflows.run_tool_step import _initialize_tool_step

    session = {"status": "confirmed", "confirmed_snapshot": {"config_version": "v", "config": {}}}
    state = {"status": "ready", "tasks": []}
    normal = _initialize_tool_step(state, session, registry={}, context={"saved_input_only": True})
    assert normal["current"] is not state
    assert "confirmed_config_version" not in state
    owned = _initialize_tool_step(
        state, session, registry={}, context={"saved_input_only": True, "_event_state_owned": True}
    )
    assert owned["current"] is state


def test_owned_result_merge_does_not_resurrect_approval():
    from phase_agent.tools.workflows.tool_outcomes import _apply_execution_result

    payload = {"tasks": [], "pending_execution_policies": {"old": {"approved": True}}}
    execution = {"status": "completed", "result": {"state": payload, "status": "completed"}}
    result, _ = _apply_execution_result(
        {"pending_execution_policies": {}},
        {"tool": "pause_search"},
        execution,
        record_id="r",
        formal=False,
        _owned=True,
    )
    assert not result["pending_execution_policies"]
    assert result["tasks"] is payload["tasks"]
    copied, _ = _apply_execution_result(
        {"pending_execution_policies": {}},
        {"tool": "pause_search"},
        execution,
        record_id="r",
        formal=False,
    )
    assert copied["tasks"] is not payload["tasks"]


def test_idle_dft_delta_isolates_mutable_ledgers_on_failure():
    from phase_agent.tools.workflows.create_dft_selection_handler import _input_working_state

    class ReadOnlyHistory:
        def __deepcopy__(self, memo):
            raise AssertionError("unrelated scientific history copied")

    source = {
        "history": ReadOnlyHistory(),
        "tasks": [{"outputs": {"energy": 1}}],
        "budget_reservations": {"old": {"status": "reserved"}},
        "slurm_batches": [{"handoff": {"folder": "old"}}],
        "upload_layout": {"operations": {"old": 1}},
        "qbc_selection_history": [],
    }
    working = _input_working_state(
        {"event_state": source, "_event_state_owned": True, "saved_input_only": True}
    )
    assert working["history"] is source["history"]
    with pytest.raises(OSError):
        working["tasks"][0]["outputs"]["energy"] = 2
        working["budget_reservations"]["old"]["status"] = "submitted"
        working["slurm_batches"][0]["handoff"]["folder"] = "new"
        working["upload_layout"]["operations"]["new"] = 2
        working["qbc_selection_history"].append({"new": True})
        raise OSError("input generation failed")
    assert source["tasks"][0]["outputs"]["energy"] == 1
    assert source["budget_reservations"]["old"]["status"] == "reserved"
    assert source["slurm_batches"][0]["handoff"]["folder"] == "old"
    assert source["upload_layout"] == {"operations": {"old": 1}}
    assert source["qbc_selection_history"] == []
    with pytest.raises(AssertionError, match="history copied"):
        _input_working_state({"event_state": source, "_event_state_owned": True})


def test_unmarked_custom_handler_state_is_not_adopted(monkeypatch):
    from phase_agent.tools.workflows.run_tool_step import _dispatch_graph_node

    external = {"tasks": []}
    monkeypatch.setattr(
        "phase_agent.tools.workflows.run_tool_step.execute_tool_action",
        lambda *a, **k: {
            "status": "completed",
            "result": {"status": "completed", "state": external},
        },
    )
    frame = {
        "current": {"tasks": []},
        "action": {"tool": "pause_search"},
        "registry": {"pause_search": {"handler": lambda **kw: None}},
        "validation": {"config_version": "v"},
        "context": {"_event_state_owned": True},
        "config": {},
        "mode": "interactive",
        "policy": {},
        "record_id": "r",
    }
    result = _dispatch_graph_node(frame)
    result["current"]["tasks"].append({"new": True})
    assert external["tasks"] == []


def test_default_dft_handler_explicitly_supports_owned_result():
    from phase_agent.tools.workflows.create_dft_selection_handler import (
        create_dft_selection_handler,
    )

    assert create_dft_selection_handler()._returns_owned_state is True


@pytest.mark.parametrize("packaging_fails", [False, True])
def test_dft_batch_builds_only_shared_upload_package(tmp_path, monkeypatch, packaging_fails):
    import tempfile
    from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("phase_agent.persistence.shared_dft.reusable_results", lambda *a, **k: [])

    def inputs(task, **kwargs):
        folder = Path(task["work_directory"])
        (folder / "initial_structure.vasp").write_text("test structure")
        (folder / "workflow.py").write_text("print('input only')")
        (folder / "submit_gpu.sh").write_text(
            "#!/bin/bash\n#SBATCH --job-name=test\npython3 workflow.py\n"
        )
        return {"generator": "atomate", "backend": "pycode_atomate2_static"}

    monkeypatch.setattr(
        "phase_agent.tools.local.prepare_dft_upload_batches.prepare_pycode_relax", inputs
    )
    if packaging_fails:
        monkeypatch.setattr(
            "phase_agent.tools.remote.submission_handoff.finalize_prepared_handoffs",
            lambda state: {"status": "not_configured", "reason": "package failure", "state": state},
        )
    state = {
        "confirmed_config_version": "v",
        "tasks": [
            {
                "task_id": f"D{i}",
                "task_key": f"dft:{i}",
                "stage": "dft_single_point",
                "status": "pending",
                "model_version": "m",
            }
            for i in range(2)
        ],
        "budget_reservations": {f"dft:{i}": {"status": "reserved"} for i in range(2)},
    }
    root = tmp_path / "submissions"
    result = prepare_dft_upload_batches(
        action={"target_ids": ["D0", "D1"]},
        context={
            "event_state": state,
            "manager": SimpleNamespace(data={}),
            "effective_config": {
                "upload_batches_directory": str(root),
                "mlip": {"version": "m", "model_path": "/HPC/model.model"},
                "python_environments": {"remote_mlip": "current"},
            },
        },
    )
    assert all(not task.get("slurm_batch_id") for task in state["tasks"])
    assert not list(root.rglob("*.upload.zip"))
    if packaging_fails:
        assert result["status"] == "not_configured"
        assert not list(root.rglob("inputs.zip"))
    else:
        assert result["status"] == "prepared", result.get("reason")
        archives = list(root.rglob("inputs.zip"))
        assert len(archives) == 1
        assert {batch["upload_archive"] for batch in result["batches"]} == {str(archives[0])}
        assert result["handoffs"][0]["job_count"] == 2
        with zipfile.ZipFile(archives[0]) as zipped:
            assert zipped.testzip() is None
            assert len([n for n in zipped.namelist() if n.endswith("/task.json")]) == 2
