"""A regrouped refresh must preserve each original result binding."""

import json
import pytest


def test_grouped_existing_batches_preserve_identity_and_isolate_failure(tmp_path, monkeypatch):
    from phase_agent.science.mlip import model_interface
    from phase_agent.tools.remote import run_mlip_batch as batch_module
    from phase_agent.tools.remote import run_single_task as single_task_module
    from phase_agent.tools.remote.integrity import verify_result

    stage = tmp_path / "epoch4" / "calculations" / "refresh_single_point"
    inputs = stage / "inputs"
    group = inputs / "regrouped"
    group.mkdir(parents=True)
    (stage / "results").mkdir()
    entries, originals = [], []
    for index in range(3):
        batch = inputs / f"old_batch_{index}"
        directory = batch / f"00000-task-{index}"
        directory.mkdir(parents=True)
        (batch / "batch.snapshot.json").write_text("{}", encoding="utf8")
        task = {
            "task_id": f"task-{index}",
            "task_key": f"key-{index}",
            "batch_id": f"old-batch-{index}",
            "config_version": "config-1",
            "model_version": "model-1",
            "task_checksum": f"checksum-{index}",
            "protocol_version": 1,
            "input_file_version": "task-json-v1",
            "stage": "relax_and_feature",
            "parameters": {"model_refresh_operation": "predict", "refresh_force_threshold": 0.1},
            "worker_job": {"operation": "predict"},
        }
        task_path = directory / "task.json"
        task_path.write_text(json.dumps(task), encoding="utf8")
        originals.append(task_path.read_bytes())
        entries.append(
            {
                **{
                    k: task[k]
                    for k in (
                        "task_id",
                        "task_key",
                        "batch_id",
                        "config_version",
                        "model_version",
                        "task_checksum",
                        "protocol_version",
                        "input_file_version",
                    )
                },
                "input_path": f"../old_batch_{index}/{directory.name}/task.json",
            }
        )
    manifest = group / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf8")
    loads, seen = [], []

    def factory(model, *, model_path=None):
        loads.append(model_path)
        return object()

    def execute(task):
        seen.append(model_interface.create_calculator({"device": "cuda"}, model_path="model"))
        if task["task_id"] == "task-1":
            raise ValueError("synthetic failure")
        return {"status": "completed", "stage": "relax_and_feature", "outputs": {}}

    monkeypatch.setattr(model_interface, "create_calculator", factory)
    monkeypatch.setattr(single_task_module, "_load", lambda _: execute)
    assert batch_module.run_mlip_batch(manifest, executor="fake:execute") == ["task-1"]
    assert loads == ["model"] and len({id(value) for value in seen}) == 1
    for index, entry in enumerate(entries):
        result = stage / "results" / f"00000-task-{index}" / "result.json"
        checked = verify_result(result, result.with_name("task.finished.json"), entry)
        assert checked["valid"], checked
        assert checked["result"]["status"] == ("failed" if index == 1 else "completed")
        assert (
            inputs / f"old_batch_{index}" / f"00000-task-{index}" / "task.json"
        ).read_bytes() == originals[index]


def test_grouped_preflight_checks_referenced_models(tmp_path):
    import pytest
    from phase_agent.tools.remote.submission_dispatcher import preflight_models

    group = tmp_path / "inputs" / "combined"
    task = tmp_path / "inputs" / "old" / "task-a"
    group.mkdir(parents=True)
    task.mkdir(parents=True)
    (group / "GPU.sh").write_text("#!/bin/bash\n", encoding="utf8")
    (group / "manifest.json").write_text(
        json.dumps([{"task_id": "task-a", "input_path": "../old/task-a/task.json"}]),
        encoding="utf8",
    )
    (task / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "stage": "relax_and_feature",
                "worker_job": {"model_path": str(tmp_path / "missing.model")},
            }
        ),
        encoding="utf8",
    )
    with pytest.raises(FileNotFoundError, match="remote_model_not_found"):
        preflight_models(tmp_path, ["inputs/combined/GPU.sh"])


def test_grouped_preflight_rejects_escaping_reference(tmp_path):
    import pytest
    from phase_agent.tools.remote.submission_dispatcher import preflight_models

    group = tmp_path / "inputs" / "combined"
    group.mkdir(parents=True)
    (group / "GPU.sh").write_text("#!/bin/bash\n", encoding="utf8")
    (group / "manifest.json").write_text(
        json.dumps([{"task_id": "task-a", "input_path": "../../../outside/task.json"}]),
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="outside"):
        preflight_models(tmp_path, ["inputs/combined/GPU.sh"])


def test_grouped_preflight_rejects_duplicate_task_identity(tmp_path):
    import pytest
    from phase_agent.tools.remote.submission_dispatcher import preflight_models

    group = tmp_path / "inputs" / "combined"
    group.mkdir(parents=True)
    (group / "GPU.sh").write_text("#!/bin/bash\n", encoding="utf8")
    rows = []
    for name in ("a", "b"):
        task = tmp_path / "inputs" / name / "task.json"
        task.parent.mkdir(parents=True)
        task.write_text(json.dumps({"task_id": "same", "worker_job": {}}), encoding="utf8")
        rows.append({"task_id": "same", "input_path": f"../{name}/task.json"})
    (group / "manifest.json").write_text(json.dumps(rows), encoding="utf8")
    with pytest.raises(ValueError, match="duplicate regrouped task"):
        preflight_models(tmp_path, ["inputs/combined/GPU.sh"])


@pytest.mark.parametrize("old_phase", [False, True])
def test_repair_repackages_registered_tasks_and_adds_phase_dependency(tmp_path, monkeypatch, old_phase):
    import hashlib
    import zipfile
    from types import SimpleNamespace
    from pathlib import Path
    from ase import Atoms
    from ase.io import write
    from phase_agent.configuration.defaults.default_slurm_cluster_config import (
        default_slurm_cluster_config,
    )
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
    from phase_agent.tools.remote.refresh_handoff_repair import rebuild_refresh_handoff
    from phase_agent.tools.remote.submission_handoff import prepare_handoff, handoff_root

    project = tmp_path / "project"
    source = tmp_path / "initial.vasp"
    write(source, Atoms("Na", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True), format="vasp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    tasks = []
    for i in range(3):
        operation = "predict" if i < 2 else "relax"
        parameters = {"model_refresh_operation": operation}
        if operation == "predict":
            parameters["refresh_force_threshold"] = 0.1
        tasks.append(
            {
                "task_id": f"id-{i}",
                "task_key": f"key-{i}",
                "stage": "relax_and_feature",
                "status": "pending",
                "branch_id": "B",
                "model_version": "model-1",
                "config_version": "config-1",
                "input_structure_sha256": digest,
                "source_path": str(source),
                "model_refresh_id": "refresh-1",
                "parent_decision_id": "approved-1",
                "parameters": parameters,
            }
        )
    state = {
        "tasks": tasks,
        "confirmed_config_version": "config-1",
        "active_model_version": "model-1",
        "budget_reservations": {t["task_key"]: {"status": "reserved"} for t in tasks},
    }

    def prepare(task):
        return {
            **task,
            "worker_job": {
                "structure_path": task["source_path"],
                "operation": task["parameters"]["model_refresh_operation"],
                "model_path": "/remote/model",
                "parameters": task["parameters"],
            },
        }

    runner = ManualUploadBatchRunner(
        project / "submissions",
        worker_command=["python", "--executor", "fake:execute"],
        task_preparer=prepare,
        stage_batch_sizes={"mlip_single_point": 1},
    )
    batches = []
    while True:
        result = runner.prepare(state, _defer_archive=True, _defer_index=True)
        state = result["state"]
        if result["status"] == "no_tasks":
            break
        batches.append(result["batch"])
    assert len(batches) == 3
    root = handoff_root(Path(batches[0]["upload_directory"]).parent.parent)
    old_scripts = [b["script_path"] for b in batches]
    old_dependencies = None
    if old_phase:
        cpu_dir = Path(batches[0]["upload_directory"]).parent / ("zz-phase-analysis-" + batches[0]["batch_id"])
        cpu_dir.mkdir()
        cpu, finalizer = cpu_dir / "CPU.sh", cpu_dir / "CPU_finalize.sh"
        cpu.write_text("#!/bin/bash\ntrue\n", encoding="utf8", newline="\n")
        finalizer.write_text("#!/bin/bash\ntrue\n", encoding="utf8", newline="\n")
        gpu_paths = [Path(p).resolve().relative_to(root.resolve()).as_posix() for p in old_scripts]
        cpu_path = cpu.resolve().relative_to(root.resolve()).as_posix()
        final_path = finalizer.resolve().relative_to(root.resolve()).as_posix()
        old_dependencies = {cpu_path: gpu_paths, final_path: [cpu_path]}
        old_scripts += [str(cpu), str(finalizer)]
    old = prepare_handoff(root, old_scripts, afterany=old_dependencies)
    old["batch_ids"] = [b["batch_id"] for b in batches]
    for b in state["slurm_batches"]:
        b["handoff"] = old
    state["model_refresh"] = {
        "status": "waiting_results",
        "approved_by": "approved-1",
        "task_ids": [t["task_id"] for t in tasks],
    }
    before = {t["task_id"]: t["task_checksum"] for t in state["tasks"]}
    old_hash = hashlib.sha256(Path(old["upload_archive"]).read_bytes()).hexdigest()
    import copy
    import pytest

    changed = copy.deepcopy(state)
    changed["tasks"][0]["job_id"] = "12345"
    with pytest.raises(ValueError, match="not_unsubmitted"):
        rebuild_refresh_handoff(
            changed,
            project,
            old_hash,
            3,
            SimpleNamespace(boundary={}, data={"system_config": {}}),
            {},
            default_slurm_cluster_config(),
        )
    result = rebuild_refresh_handoff(
        state,
        project,
        old_hash,
        3,
        SimpleNamespace(boundary={}, data={"system_config": {}}),
        {},
        default_slurm_cluster_config(),
    )
    repeated = rebuild_refresh_handoff(
        state,
        project,
        old_hash,
        3,
        SimpleNamespace(boundary={}, data={"system_config": {}}),
        {},
        default_slurm_cluster_config(),
    )
    assert repeated["status"] == "reused"
    assert len(state["refresh_handoff_repairs"]) == 1
    assert result["gpu_job_count"] == 2
    assert result["prediction_compatibility_counts"] == [2]
    assert {t["task_id"]: t["task_checksum"] for t in state["tasks"]} == before
    assert Path(old["upload_archive"]).is_file()
    assert Path(result["handoff"]["return_directory"]).is_dir()
    with zipfile.ZipFile(result["handoff"]["upload_archive"]) as archive:
        plan = json.loads(archive.read("submission_plan.json"))
        assert len(plan["scripts"]) == 3
        cpu, parents = next(iter(plan["afterany"].items()))
        assert cpu.endswith("CPU.sh") and len(parents) == 2
        wave = next(n for n in archive.namelist() if n.endswith("phase_wave.json"))
        assert len(json.loads(archive.read(wave))["tasks"]) == 3
        assert (
            sum(name.endswith("/_handoff_refresh_executor.py") for name in archive.namelist()) == 1
        )
        remote = tmp_path / "remote"
        archive.extractall(remote)
        group = next(name for name in plan["scripts"] if "/regroup-" in name)
    from phase_agent.science.mlip import model_interface
    from phase_agent.tools.remote import run_mlip_batch as batch_module
    from phase_agent.tools.remote import run_single_task as single_task_module

    loaded = []
    monkeypatch.setattr(
        model_interface, "create_calculator", lambda *a, **k: loaded.append(1) or object()
    )

    def execute(task):
        model_interface.create_calculator({}, model_path="fake")
        return {"status": "completed", "stage": "relax_and_feature", "outputs": {}}

    monkeypatch.setattr(single_task_module, "_load", lambda _: execute)
    manifest = (remote / group).parent / "manifest.json"
    assert batch_module.run_mlip_batch(manifest, executor="fake:execute") == []
    assert loaded == [1]
    assert len(list((remote / "Relax-screening" / "results").glob("*/result.json"))) == 2


def test_repair_control_rejects_ledger_change_and_reuse_does_not_write(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import pytest
    from phase_agent.runtime import refresh_handoff_repair_control as control_module
    from phase_agent.tools.remote import refresh_handoff_repair as repair_module
    from phase_agent.tools.state import execution_receipts

    path = tmp_path / "workflow_state" / "state.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf8")
    manager = SimpleNamespace(_ledger_path=str(tmp_path / "ledger.json"))
    control = SimpleNamespace(
        state_path=path,
        _codex_context_cache=None,
        chat_handler=SimpleNamespace(lock=threading.RLock(), workflow_kwargs={"manager": manager}),
    )
    writes = []
    monkeypatch.setattr(control_module, "read_identity", lambda *a, **k: "version")
    monkeypatch.setattr(control_module, "read_json", lambda *a: {})
    monkeypatch.setattr(control_module, "write_json_if_version", lambda *a: writes.append(a))
    monkeypatch.setattr(control_module, "refresh_if_changed", lambda *a: None)
    monkeypatch.setattr(execution_receipts, "recovery_report", lambda *a: {"unsettled": []})
    versions = iter(["before", "after"])
    monkeypatch.setattr(control_module, "_ledger_identity", lambda _: next(versions))
    monkeypatch.setattr(repair_module, "rebuild_refresh_handoff", lambda *a: {"status": "prepared"})
    with pytest.raises(ValueError, match="ledger_changed"):
        control_module.repair(control, "a" * 64, 3)
    assert not writes
    monkeypatch.setattr(control_module, "_ledger_identity", lambda _: "same")
    monkeypatch.setattr(repair_module, "rebuild_refresh_handoff", lambda *a: {"status": "reused"})
    assert control_module.repair(control, "a" * 64, 3)["status"] == "reused"
    assert not writes


def test_mcp_repair_route_requires_exact_arguments_and_delegates():
    from types import SimpleNamespace
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    calls = []
    control = SimpleNamespace(
        repair_refresh_handoff=lambda **args: calls.append(args) or {"status": "prepared"}
    )
    server = CodexMcpServer(SimpleNamespace(control=control))
    payload = {"source_sha256": "a" * 64, "task_count": 3}
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "phase_repair_refresh_handoff", "arguments": payload},
        }
    )
    assert json.loads(response["result"]["content"][0]["text"])["status"] == "prepared"
    assert calls == [payload]
    payload["extra"] = True
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "phase_repair_refresh_handoff", "arguments": payload},
        }
    )
    assert "error" in response and len(calls) == 1


def test_mcp_repair_uses_saved_view_without_initializing_runtime(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.runtime import saved_readonly_project
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text("{}", encoding="utf8")
    calls = []
    control = SimpleNamespace(
        repair_refresh_handoff=lambda **args: calls.append(args) or {"status": "prepared"}
    )
    monkeypatch.setattr(
        saved_readonly_project, "SavedReadOnlyProject", lambda _: SimpleNamespace(control=control)
    )
    server = CodexMcpServer(runtime_config=runtime)
    result = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "phase_repair_refresh_handoff",
                "arguments": {"source_sha256": "a" * 64, "task_count": 3},
            },
        }
    )
    assert json.loads(result["result"]["content"][0]["text"])["status"] == "prepared"
    assert calls[0]["runtime_config_path"] == str(runtime.resolve()) and server._needs_runtime
