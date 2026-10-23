"""Small immutable archives reproduce a renamed local recovery delivery."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phase_agent.tools.remote.legacy_recovery import recovery_handoffs, recovery_plan
from phase_agent.tools.remote.transfer_bundle import create_bundle
from phase_agent.runtime.status_presentation import format_progress
from phase_agent.runtime.project_diagnostics import inspect_project


def setup_delivery(tmp_path, *, registered=True):
    source = tmp_path / "submissions/epoch0/transfers/DFTSP_Batch01"
    original = tmp_path / "submissions/archives/01-upload/epoch0_DFTSP_Batch01"
    source.mkdir(parents=True)
    payload = tmp_path / "payload"
    directory = payload / "operation/inputs/submission/task"
    directory.mkdir(parents=True)
    task = {
        "task_id": "DFT-one",
        "task_key": "key-one",
        "stage": "dft_single_point",
        "status": "completed",
        "batch_id": "b1",
        "config_version": "c1",
        "model_version": "m1",
        "task_checksum": "task-sha",
        "protocol_version": 1,
        "input_file_version": 1,
    }
    (directory / "task.json").write_text(json.dumps(task), encoding="utf-8")
    (payload / "submission_plan.json").write_text(
        json.dumps({"scripts": ["operation/inputs/submission/task/GPU.sh"]}), encoding="utf-8"
    )
    create_bundle(
        payload,
        [directory / "task.json", payload / "submission_plan.json"],
        source / "inputs.zip",
        kind="inputs",
    )
    handoff = {
        "folder": str(source),
        "upload_archive": str(source / "inputs.zip"),
        "import_directory": str(tmp_path / "submissions/epoch0"),
    }
    state = {
        "active_model_version": "m1",
        "tasks": [task, {"task_id": "MC-one", "stage": "deep_search", "status": "pending"}],
        "slurm_batches": [{"batch_id": "b1", "task_ids": [task["task_id"]], "handoff": handoff}],
        "confirmed_config": {
            "storage": {
                "workspace_root": str(tmp_path),
                "paths": {"state": "workflow_state/state.json"},
            }
        },
    }
    plan = recovery_plan(state, handoff)
    plan["source_folder"] = original.name
    recovery = source.with_name(source.name + "_Recovery01_input")
    scratch = tmp_path / "recovery-payload"
    scratch.mkdir()
    plan_file = scratch / "recovery_plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    create_bundle(scratch, [plan_file], recovery / "inputs.zip", kind="inputs")
    if registered:
        state["path_relocation_revision"] = "move-one"
        control = tmp_path / "workflow_state/state.json.control.json"
        control.parent.mkdir(parents=True)
        control.write_text(
            json.dumps(
                {
                    "state_path": str((control.parent / "state.json").resolve()),
                    "revision": "r1",
                    "path_relocation_revision": "move-one",
                    "path_relocations": [{"from": str(original), "to": str(source)}],
                }
            ),
            encoding="utf-8",
        )
    return state, handoff, plan_file, recovery


def test_exact_registered_migration_preserves_original_remote_folder(tmp_path):
    state, handoff, _, recovery = setup_delivery(tmp_path)
    before = deepcopy(state)
    assert recovery_plan(state, handoff)["source_folder"] == "epoch0_DFTSP_Batch01"
    deliveries = list(recovery_handoffs(state))
    assert len(deliveries) == 1 and deliveries[0]["folder"] == str(recovery)
    assert deliveries[0]["task_ids"] == ["DFT-one"]
    assert state == before


def test_status_and_readiness_check_the_real_waiting_reply(tmp_path, monkeypatch):
    state, _, _, _ = setup_delivery(tmp_path)
    before = deepcopy(state)
    reply = format_progress(state)
    assert "MC" in reply and state == before
    path = tmp_path / "workflow_state/state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    assert inspect_project(path)["sections"]["status_presentation"]["data"]["valid"]

    def failing(_):
        raise ValueError("recovery_delivery_outside_original_scope")

    monkeypatch.setattr("phase_agent.runtime.status_presentation.format_progress", failing)
    report = inspect_project(path)
    assert not report["sections"]["status_presentation"]["available"]
    assert any(row["object_id"] == "status_presentation" for row in report["issues"])


@pytest.mark.parametrize("field", ["tasks", "source_archive_sha256", "mode", "source_folder"])
def test_registered_migration_still_rejects_changed_scope(tmp_path, field):
    state, _, plan_file, recovery = setup_delivery(tmp_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if field == "tasks":
        plan[field][0]["expected"]["task_checksum"] = "forged"
    else:
        plan[field] = "forged"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    create_bundle(plan_file.parent, [plan_file], recovery / "inputs.zip", kind="inputs")
    with pytest.raises(ValueError, match="outside_original_scope"):
        list(recovery_handoffs(state))


@pytest.mark.parametrize("registered,revision", [(False, None), (True, "unknown")])
def test_unregistered_or_stale_folder_alias_is_not_accepted(tmp_path, registered, revision):
    state, _, _, _ = setup_delivery(tmp_path, registered=registered)
    state["path_relocation_revision"] = revision
    with pytest.raises(ValueError, match="outside_original_scope"):
        list(recovery_handoffs(state))


def test_relocated_reassessment_keeps_verified_cache_and_archive_binding(tmp_path):
    from hashlib import sha256
    from phase_agent.tools.remote.transfer_bundle import registered_return_locations

    state, handoff, plan_file, recovery = setup_delivery(tmp_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    remote_name = "epoch0_DFTSP_Batch01_Recovery01_input"
    key = sha256((remote_name + plan["source_archive_sha256"]).encode()).hexdigest()[:12]
    existing = tmp_path / "submissions/epoch0/recovery" / key
    existing.mkdir(parents=True)
    marker = plan_file.with_name("repair_options.json")
    marker.write_text("{}", encoding="utf-8")
    create_bundle(plan_file.parent, [plan_file, marker], recovery / "inputs.zip", kind="inputs")
    control = tmp_path / "workflow_state/state.json.control.json"
    data = json.loads(control.read_text(encoding="utf-8"))
    data["path_relocations"].append(
        {
            "from": str(tmp_path / "submissions/archives/01-upload" / remote_name),
            "to": str(recovery),
        }
    )
    control.write_text(json.dumps(data), encoding="utf-8")
    original_archive = (recovery / "inputs.zip").read_bytes()
    handoffs = list(recovery_handoffs(state))
    assert handoffs[0]["same_frame_prediction"] is True
    assert Path(handoffs[0]["import_directory"]) == existing
    locations = registered_return_locations(state)
    target = locations[str(recovery.with_name(recovery.name[:-6] + "_output"))]
    assert target["directory"] == str(existing)
    assert target["batch"] == remote_name[:-6]
    assert (recovery / "inputs.zip").read_bytes() == original_archive


def test_live_workspace_binding_precedes_frozen_config_storage(tmp_path):
    state, _, _, _ = setup_delivery(tmp_path)
    state["workspace_layout"] = {
        "workspace_root": str(tmp_path),
        "state_path": str(tmp_path / "workflow_state/state.json"),
    }
    state["confirmed_config"]["storage"] = {"workspace_root": str(tmp_path / "old-workspace")}
    assert len(list(recovery_handoffs(state))) == 1
