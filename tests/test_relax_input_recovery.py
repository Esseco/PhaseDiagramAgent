from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.tools.local.prepare_relax_upload_batches import prepare_relax_upload_batches


def context(tmp_path):
    source = tmp_path / "original.vasp"
    source.write_text("unchanged original scientific input", encoding="utf-8")
    root = tmp_path / "submissions"
    batch = root / "operation/inputs/batch-1"
    key = (
        "relax-screen:m:"
        + hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()[:12]
        + ":s"
    )
    task = {
        "task_id": "R1",
        "task_key": key,
        "branch_id": "b",
        "structure_id": "s",
        "stage": "relax_and_feature",
        "status": "pending",
        "slurm_batch_id": "batch-1",
        "batch_id": "batch-1",
        "input_path": str(batch / "00000-R1/task.json"),
        "task_checksum": "original-checksum",
    }
    manager = SimpleNamespace(
        data={
            "branches": {"b": {"structure_ids": ["s"]}},
            "structures": {
                "s": {
                    "branch_id": "b",
                    "source_path": str(source),
                    "composition": {"Na": 1, "Fe": 1, "O": 2},
                    "metadata": {
                        "initialization_method": "electrostatic_top10_random3_layer_occupied"
                    },
                }
            },
        }
    )
    config = {
        "mlip": {"version": "m", "model_path": "/remote/model"},
        "budgets": default_budget_rules(),
        "upload_batches_directory": str(root),
        "supercomputer": {
            "worker": {
                "command": [
                    "python3",
                    "--executor",
                    "phase_agent.science.mlip.slurm_executor:execute_mlip_task",
                ]
            }
        },
    }
    return (
        {
            "effective_config": config,
            "manager": manager,
            "config_version": "c",
            "event_state": {
                "tasks": [task],
                "dedup_gate": {"status": "ready"},
                "slurm_batches": [{"batch_id": "batch-1", "upload_directory": str(batch)}],
            },
        },
        task,
        batch,
    )


@pytest.mark.parametrize("status", ["pending", "running", "submitted", "unknown", "cancelled"])
def test_missing_bound_inputs_never_reset_job_identity(tmp_path, status):
    ctx, task, _ = context(tmp_path)
    task["status"] = status
    if status == "cancelled":
        task["recovery_cancelled"] = True
    original = deepcopy(ctx["event_state"])
    result = prepare_relax_upload_batches(action={"parameters": {}}, context=ctx)
    assert result["status"] == "not_configured"
    assert result["state"]["tasks"] == original["tasks"]
    assert ctx["event_state"] == original
    assert result["task_ids"] == ["R1"] and result["recovery"]


def test_deleted_local_inputs_restore_from_verified_archive_without_repreparing(
    tmp_path, monkeypatch
):
    from phase_agent.tools.remote.submission_storage import (
        pack_storage_directory,
        prune_storage_directory,
    )

    ctx, task, batch = context(tmp_path)
    directory = batch / "00000-R1"
    directory.mkdir(parents=True)
    for path, data in [
        (directory / "task.json", "{}"),
        (directory / "initial.vasp", "original geometry"),
        (batch / "run_mlip_batch.py", "# original frozen runner"),
        (batch / "GPU.sh", "#!/bin/bash\nexit 0\n"),
    ]:
        path.write_text(data, encoding="utf-8")
    pack_storage_directory(batch.parent, "inputs")
    prune_storage_directory(batch.parent, "inputs")
    assert not Path(task["input_path"]).exists()
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_handoff.finalize_prepared_handoffs",
        lambda state: {"status": "already_prepared"},
    )
    monkeypatch.setattr(
        "phase_agent.tools.remote.write_relax_upload_plan.write_relax_upload_plan",
        lambda *args: None,
    )
    result = prepare_relax_upload_batches(action={"parameters": {}}, context=ctx)
    assert result["status"] == "already_prepared"
    assert result["state"]["tasks"][0] == task
    assert Path(task["input_path"]).read_text(encoding="utf-8") == "{}"
    assert (directory / "initial.vasp").read_text(encoding="utf-8") == "original geometry"


@pytest.mark.parametrize("fault", ["corrupt_zip", "manifest_list", "bad_entries"])
def test_invalid_archive_is_local_recovery_issue_and_keeps_task_binding(tmp_path, fault):
    import zipfile

    ctx, task, batch = context(tmp_path)
    archive = batch.parent / "all.inputs.zip"
    archive.parent.mkdir(parents=True)
    if fault == "corrupt_zip":
        archive.write_bytes(b"interrupted transfer")
    else:
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(
                "transfer_manifest.json",
                json.dumps(
                    []
                    if fault == "manifest_list"
                    else {"schema": 1, "kind": "storage_inputs", "files": [None]}
                ),
            )
    original = deepcopy(ctx["event_state"])
    result = prepare_relax_upload_batches(action={"parameters": {}}, context=ctx)
    assert result["status"] == "not_configured"
    assert result["state"]["tasks"] == original["tasks"] and ctx["event_state"] == original
    assert "恢复" in result["reason"]
