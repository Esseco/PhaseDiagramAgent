import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
from ase.io import write

from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.remote import run_mlip_batch as batch_module


def test_pure_refresh_batch_shares_one_helper_set(tmp_path):
    structure = tmp_path / "source.vasp"
    write(structure, Atoms("Na", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True), format="vasp")
    digest = hashlib.sha256(structure.read_bytes()).hexdigest()
    params = {"model_refresh_operation": "predict", "refresh_force_threshold": 0.1}
    tasks = [
        {
            "task_id": f"id-{i}",
            "task_key": f"key-{i}",
            "stage": "relax_and_feature",
            "status": "pending",
            "branch_id": "B",
            "model_version": "model-1",
            "config_version": "config-1",
            "input_structure_sha256": digest,
            "source_path": str(structure),
            "model_refresh_id": "refresh-1",
            "parent_decision_id": "approved-1",
            "parameters": params,
        }
        for i in range(2)
    ]
    state = {
        "tasks": tasks,
        "active_model_version": "model-1",
        "confirmed_config_version": "config-1",
        "budget_reservations": {t["task_key"]: {"status": "reserved"} for t in tasks},
    }

    def prepare(task):
        return {
            **task,
            "worker_job": {
                "structure_path": task["source_path"],
                "operation": "predict",
                "model_path": "/remote/model",
                "parameters": params,
            },
        }

    runner = ManualUploadBatchRunner(
        tmp_path / "submissions",
        worker_command=["python", "--executor", "fake:execute"],
        task_preparer=prepare,
        stage_batch_sizes={"mlip_single_point": 2},
    )
    result = runner.prepare(state, _defer_archive=True, _defer_index=True)
    folder = Path(result["batch"]["upload_directory"])
    rows = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    assert len(rows) == 2
    directories = [(folder / row["input_path"]).parent for row in rows]
    for name in (
        "_handoff_refresh_executor.py",
        "_handoff_refresh_worker.py",
        "_handoff_refresh_inline.py",
    ):
        assert (directories[0] / name).is_file()
        assert not (directories[1] / name).exists()


def test_invalid_task_fallback_still_finds_shared_helpers(tmp_path, monkeypatch):
    batch = tmp_path / "inputs" / "batch"
    first, second = batch / "first", batch / "second"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "_handoff_refresh_executor.py").write_text("# shared\n")
    (first / "task.json").write_text("invalid json")
    (second / "task.json").write_text("{}")
    manifest = batch / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"task_id": "bad", "input_path": "first/task.json"},
                {"task_id": "good", "input_path": "second/task.json"},
            ]
        )
    )
    seen = []

    def run(command, *, cwd, env, **kwargs):
        seen.append(env["PYTHONPATH"].split(os.pathsep)[0])
        if cwd == first:
            return SimpleNamespace(returncode=1)
        (cwd / "task.finished.json").write_text('{"status":"completed"}')
        (cwd / "result.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch_module.subprocess, "run", run)
    monkeypatch.setattr(batch_module, "export_batch_result", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.result_bundle", lambda *a, **k: None
    )
    failures = batch_module.run_mlip_batch(manifest, executor="_handoff_refresh_executor:execute")
    assert failures == ["bad"]
    assert seen == [str(first.resolve()), str(first.resolve())]
