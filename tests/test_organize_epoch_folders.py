import json
import os
from pathlib import Path
from copy import deepcopy

import pytest

from phase_agent.tools.local.organize_epoch_folders import folder_plan, migrate_folders
from phase_agent.runtime.interaction_report_paths import report_directory
from phase_agent.tools.remote.model_upload_directory import model_upload_directory
from phase_agent.tools.remote.submission_handoff import handoff_root


def state():
    return {
        "active_model_version": "model-long-id",
        "upload_layout": {"model_rounds": {"model-long-id": 2}},
    }


def test_new_directories_use_recorded_epoch_and_keep_identity(tmp_path):
    saved = state()
    before = deepcopy(saved)
    assert model_upload_directory(tmp_path, saved, "model-long-id") == tmp_path / "epoch1"
    assert report_directory(tmp_path, saved) == tmp_path / "interaction_records/epoch1"
    assert saved == before
    assert handoff_root(tmp_path / "epoch1/G0001_DFT-0001/inputs") == tmp_path / "epoch1"
    assert report_directory(tmp_path, {"active_model_version": "unknown"}).name == "unassigned"
    with pytest.raises(ValueError, match="ambiguous"):
        model_upload_directory(tmp_path, {"upload_layout": {"model_rounds": {"a": 1, "b": 1}}}, "a")


def test_new_reassessment_results_use_epoch_and_keep_existing_recovery(tmp_path):
    from phase_agent.tools.remote.legacy_recovery import _recovery_import_directory

    original = {"import_directory": str(tmp_path / "submissions/epoch1/G0001_DFT-0001")}
    assert (
        _recovery_import_directory(original, "key") == tmp_path / "submissions/epoch1/recovery/key"
    )
    legacy = Path(original["import_directory"]).parent / "_recovery/key"
    legacy.mkdir(parents=True)
    assert _recovery_import_directory(original, "key") == legacy
    unscoped = {"import_directory": str(tmp_path / "unregistered")}
    assert _recovery_import_directory(unscoped, "key") == tmp_path / "_recovery/key"


def test_collect_only_index_has_no_dead_upload_link(tmp_path):
    from phase_agent.tools.remote.submission_layout import write_submission_index

    root = tmp_path / "submissions"
    directory = root / "epoch1/recovery/collect_only"
    directory.with_name("collect_only_output").mkdir(parents=True)
    saved = state()
    saved["relax_mc_batches"] = {
        "b": {
            "bundle_id": "b",
            "directory": str(directory),
            "model_version": "model-long-id",
            "status": "completed",
        }
    }
    text = write_submission_index(root, saved).read_text(encoding="utf-8")
    assert "无上传目录" in text
    assert "collect_only_input](" not in text
    assert "](epoch1/recovery/collect_only_output/)" in text


def test_missing_historical_handoff_is_marked_without_dead_link(tmp_path):
    from phase_agent.tools.remote.submission_layout import write_submission_index

    root = tmp_path / "submissions"
    operation = root / "epoch1/G0001_DFT-0001"
    operation.mkdir(parents=True)
    saved = state()
    saved["slurm_batches"] = [
        {
            "upload_directory": str(operation / "inputs/batch"),
            "handoff": {
                "folder": str(root / "epoch1/transfers/missing_input"),
                "results_directory": str(operation / "results"),
                "job_count": 1,
            },
        }
    ]
    text = write_submission_index(root, saved).read_text(encoding="utf-8")
    assert "上传包目录缺失，先核对归档" in text
    assert "missing_input](" not in text


@pytest.mark.skipif(os.name != "nt", reason="Windows junction migration")
def test_direct_migration_preserves_bytes_removes_old_paths_and_is_repeatable(tmp_path):
    old = tmp_path / "analysis_outputs/epoch1_model-long-id"
    old.mkdir(parents=True)
    payload = b'{"energy":-12.5,"frozen_path":"unchanged"}'
    (old / "result.json").write_bytes(payload)
    reports = tmp_path / "交互记录"
    reports.mkdir()
    (reports / "epoch1_DFT选点_20261020_01.md").write_text("saved reply", encoding="utf-8")
    before = state()
    result = migrate_folders(tmp_path, before, apply=True)
    new = tmp_path / "analysis_outputs/epoch1"
    assert (new / "result.json").read_bytes() == payload
    assert not old.exists()
    assert not new.is_junction()
    assert not new.stat().st_file_attributes & 2
    assert (
        tmp_path / "interaction_records/epoch1/DFT选点_20261020_01.md"
    ).read_text() == "saved reply"
    assert folder_plan(tmp_path, before) == []
    assert result["verified_files"] == 2
    assert before == state()


def test_conflicting_destination_aborts_before_any_move(tmp_path):
    for name in ["epoch1_model-long-id", "epoch1"]:
        (tmp_path / "analysis_outputs" / name).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        migrate_folders(tmp_path, state(), apply=True)
    assert (tmp_path / "analysis_outputs/epoch1_model-long-id").is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction migration")
def test_storage_index_survives_epoch_migration(tmp_path):
    from phase_agent.tools.remote.submission_storage import storage_archive

    root = tmp_path / "submissions"
    (root / "epoch1_model-long-id/G0001/inputs").mkdir(parents=True)
    archive = root / "archives/01-upload/epoch1__G0001.inputs.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"frozen zip content")
    index = {"epoch1_model-long-id/G0001/inputs": "archives/01-upload/epoch1__G0001.inputs.zip"}
    (root / ".archive_index.json").write_text(json.dumps(index))
    import ctypes

    ctypes.windll.kernel32.SetFileAttributesW(str(root / ".archive_index.json"), 0x22)
    migrate_folders(tmp_path, state(), apply=True)
    assert (
        storage_archive(root / "epoch1/G0001/inputs", "inputs")
        == root / "epoch1/storage_archives/G0001.inputs.zip"
    )
    assert not archive.exists()
    assert (root / "epoch1/storage_archives/G0001.inputs.zip").read_bytes() == b"frozen zip content"
    assert (root / ".archive_index.json").stat().st_file_attributes & 2


@pytest.mark.skipif(os.name != "nt", reason="Windows junction migration")
def test_index_publish_failure_still_rolls_back_directories(tmp_path, monkeypatch):
    from phase_agent.tools.local import organize_epoch_folders as module

    old = tmp_path / "submissions/epoch1_model-long-id"
    old.mkdir(parents=True)
    (old / "input.json").write_bytes(b"unchanged input")
    index = tmp_path / "submissions/.archive_index.json"
    index.write_text("{}")

    def fail(*args):
        raise OSError("index write failed")

    monkeypatch.setattr(module, "_write_index_bytes", fail)
    with pytest.raises(OSError, match="index write failed"):
        module.migrate_folders(tmp_path, state(), apply=True)
    assert not old.is_junction()
    assert (old / "input.json").read_bytes() == b"unchanged input"
    assert not (tmp_path / "submissions/epoch1").exists()
    assert index.read_text() == "{}"


def test_cleanup_removes_only_identical_archived_runtime(tmp_path):
    import zipfile
    from phase_agent.tools.local.organize_epoch_folders import prune_archived_runtime

    batch = tmp_path / "submissions/epoch0/relax_mc/RelaxMC_Batch01"
    (batch / "runtime").mkdir(parents=True)
    (batch / "runtime/shared.py").write_bytes(b"same source")
    (batch / "runtime/changed.py").write_bytes(b"unique source")
    (batch / "result.json").write_bytes(b"scientific record")
    with zipfile.ZipFile(batch / "inputs.zip", "w") as zipped:
        zipped.writestr("runtime/shared.py", b"same source")
        zipped.writestr("runtime/changed.py", b"different source")
    plan = prune_archived_runtime(tmp_path)
    assert len(plan[0]["files"]) == 1
    assert (batch / "runtime/shared.py").exists()
    prune_archived_runtime(tmp_path, apply=True)
    assert not (batch / "runtime/shared.py").exists()
    assert (batch / "runtime/changed.py").read_bytes() == b"unique source"
    assert (batch / "result.json").read_bytes() == b"scientific record"
    assert (batch / "inputs.zip").is_file()
    assert prune_archived_runtime(tmp_path) == []


def test_legacy_package_epoch_comes_from_frozen_tasks(tmp_path):
    import zipfile

    old = tmp_path / "submissions/archives/01-upload/G0001_DFT__opaque"
    old.mkdir(parents=True)
    with zipfile.ZipFile(old / "inputs.zip", "w") as zipped:
        zipped.writestr("inputs/D1/task.json", json.dumps({"model_version": "model-long-id"}))
    assert folder_plan(tmp_path, state()) == [
        (old, tmp_path / "submissions/epoch1/transfers/G0001_DFT_legacy_package")
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction migration")
def test_recovery_analyses_are_grouped_by_their_recorded_epoch(tmp_path):
    old = tmp_path / "analysis_outputs/recovery/digest/epoch1_model-long-id"
    old.mkdir(parents=True)
    (old / "analysis.json").write_bytes(b"preserved recovery")
    current = tmp_path / "analysis_outputs/epoch1_model-long-id"
    current.mkdir()
    (current / "current.csv").write_bytes(b"preserved current")
    report = migrate_folders(tmp_path, state(), apply=True)
    assert report["verified_files"] == 2
    target = tmp_path / "analysis_outputs/epoch1/recovery/digest/analysis.json"
    assert target.read_bytes() == b"preserved recovery"
    assert not old.exists()
    assert folder_plan(tmp_path, state()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows junction migration")
def test_cleanup_counts_compatibility_paths_only_once(tmp_path):
    import zipfile
    from phase_agent.tools.local.organize_epoch_folders import prune_archived_runtime

    batch = tmp_path / "submissions/epoch1_model-long-id/relax_mc/RelaxMC_Batch01"
    (batch / "runtime").mkdir(parents=True)
    (batch / "runtime/code.py").write_bytes(b"same source")
    with zipfile.ZipFile(batch / "inputs.zip", "w") as zipped:
        zipped.writestr("runtime/code.py", b"same source")
    migrate_folders(tmp_path, state(), apply=True)
    plan = prune_archived_runtime(tmp_path)
    assert len(plan) == 1 and len(plan[0]["files"]) == 1


def test_occupied_directory_fails_without_creating_a_link(tmp_path, monkeypatch):
    old = tmp_path / "analysis_outputs/epoch1_model-long-id"
    old.mkdir(parents=True)
    (old / "result.csv").write_bytes(b"energy\n-1.23\n")
    original = Path.rename

    def busy(path, target):
        if path == old:
            raise PermissionError("occupied directory")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", busy)
    with pytest.raises(PermissionError, match="occupied"):
        migrate_folders(tmp_path, state(), apply=True)
    assert not (tmp_path / "analysis_outputs/epoch1").exists()
    assert (old / "result.csv").read_bytes() == b"energy\n-1.23\n"
    assert not old.is_junction()


def test_chained_relocations_rebind_operational_fields_without_changing_frozen_config(tmp_path):
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths

    old = tmp_path / "old"
    intermediate = tmp_path / "submissions/epoch1_model"
    new = tmp_path / "submissions/epoch1"
    moves = [
        {"from": str(old), "to": str(intermediate)},
        {"from": str(intermediate), "to": str(new)},
    ]
    value = {
        "tasks": [{"path": str(old / "result.json"), "energy": -1.23}],
        "confirmed_config": {"path": str(old / "frozen")},
    }
    result = relocate_workspace_paths(value, moves, tmp_path, preserve_snapshots=True)
    assert result["tasks"] == [{"path": str(new / "result.json"), "energy": -1.23}]
    assert result["confirmed_config"] == value["confirmed_config"]
    assert value["tasks"][0]["path"] == str(old / "result.json")
    assert relocate_workspace_paths(result, moves, tmp_path, preserve_snapshots=True) == result
    with pytest.raises(ValueError, match="循环"):
        relocate_workspace_paths(
            str(old),
            [{"from": str(old), "to": str(new)}, {"from": str(new), "to": str(old)}],
            tmp_path,
        )


def test_checkpoint_paths_rebound_even_when_control_revision_is_current(tmp_path):
    from phase_agent.tools.state.workflow_control import restore_control, control_path

    state_path = tmp_path / "workflow_state/state.json"
    state_path.parent.mkdir()
    old, new = tmp_path / "submissions/epoch1_model", tmp_path / "submissions/epoch1"
    record = {
        "state_path": str(state_path.resolve()),
        "revision": "r",
        "path_relocation_revision": "paths-r",
        "path_relocations": [{"from": str(old), "to": str(new)}],
    }
    control_path(state_path).write_text(json.dumps(record))
    saved = {
        "workflow_control_revision": "r",
        "tasks": [{"directory": str(old)}],
        "confirmed_config": {"path": str(old)},
    }
    rebound = restore_control(state_path, saved)
    assert rebound["tasks"][0]["directory"] == str(new)
    assert rebound["confirmed_config"] == saved["confirmed_config"]
    assert saved["tasks"][0]["directory"] == str(old)


def test_expanding_path_mapping_is_rejected_instead_of_looping(tmp_path):
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths

    old = tmp_path / "old"
    with pytest.raises(ValueError, match="循环"):
        relocate_workspace_paths(
            str(old / "result.json"), [{"from": str(old), "to": str(old / "child")}], tmp_path
        )
