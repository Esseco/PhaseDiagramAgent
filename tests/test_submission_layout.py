import json
from pathlib import Path
import pytest
from phase_agent.tools.remote.build_upload_batch_directory import (
    build_upload_batch_directory,
)
from phase_agent.tools.remote.submission_layout import (
    compact_calculation_directory,
    submission_moves,
)
from phase_agent.tools.remote.migrate_submission_layout import migrate_submission_layout
from phase_agent.tools.remote.batch_runner import _task_result_path


@pytest.mark.parametrize(
    "stage,extra,label",
    [
        ("relax_and_feature", {}, "G0001_Relax-0001"),
        (
            "deep_search",
            {"segment_index": 1, "parent_relax_round": 2},
            "G0001_Relax-0002_MC-round-0002",
        ),
        ("dft_single_point", {}, "G0001_DFT-0001_DFT-single-point"),
        ("dft_relax", {}, "G0001_DFT-0001_DFT-relax"),
    ],
)
def test_flat_builder_preserves_batch_and_sibling_results(
    tmp_path, stage, extra, label
):
    directory, metadata = build_upload_batch_directory(
        tmp_path,
        {},
        batch_id="remote-000001",
        stage=stage,
        model_version="m1",
        operation_id="operation",
        search_group_index=1,
        **extra,
    )
    assert directory.parent.parent == tmp_path / "epoch0" / label
    assert directory.parent.name == "inputs"
    assert directory.name.endswith("remote-000001")
    assert metadata["submission_layout_version"] == 2
    assert (
        compact_calculation_directory(tmp_path, directory.parent.parent)
        == directory.parent.parent
    )


def test_groups_and_refresh_do_not_collide(tmp_path):
    first = tmp_path / "epoch0_m1/Search-group-0001/Relax-0001"
    second = tmp_path / "epoch0_m1/Search-group-0002/Relax-0001"
    assert compact_calculation_directory(
        tmp_path, first
    ) != compact_calculation_directory(tmp_path, second)
    directory, _ = build_upload_batch_directory(
        tmp_path,
        {},
        batch_id="remote-000001",
        stage="relax_and_feature",
        model_version="m1",
        model_refresh_round=1,
        refresh_operation="relax",
    )
    assert directory.parent.parent.name == "Model-refresh-0001_Relax"


def test_migration_preserves_payload_and_historical_approval(tmp_path):
    project = tmp_path / "project"
    old = project / "submissions/epoch0_m1/Search-group-0001/Relax-0001"
    batch = old / "inputs/Relax-submission-0001_remote-000001"
    batch.mkdir(parents=True)
    frozen = b'{"old_path":"untouched", "task_checksum":"frozen"}'
    (batch / "task.json").write_bytes(frozen)
    state_path = project / "workflow_state/state.json"
    state_path.parent.mkdir()
    approval = {"structure_path": str(old / "results/a.vasp"), "proposal_hash": "keep"}
    state = {
        "slurm_batches": [
            {"upload_directory": str(batch), "results_directory": str(old / "results")}
        ],
        "tasks": [{"input_path": str(batch / "task.json")}],
        "pending_execution_policies": approval,
        "confirmed_config_version": "same",
        "active_model_version": "same",
    }
    state_path.write_text(json.dumps(state))
    result = migrate_submission_layout(project, tmp_path / "backup")
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    new_batch = Path(updated["slurm_batches"][0]["upload_directory"])
    assert new_batch.parent.parent.name == "G0001_Relax-0001"
    assert (new_batch / "task.json").read_bytes() == frozen
    assert not old.exists()
    assert not old.is_junction() and not old.is_symlink()
    assert updated["pending_execution_policies"] == approval
    assert (
        updated["confirmed_config_version"] == updated["active_model_version"] == "same"
    )
    assert result["immutable_files_verified"] == 1
    assert submission_moves(project / "submissions", updated) == []
    assert (
        migrate_submission_layout(project, tmp_path / "backup")["status"]
        == "already_compact"
    )


def test_rejects_outside_or_collision_before_mutation(tmp_path):
    old = tmp_path / "epoch0_m1/Search-group-0001/Relax-0001"
    old.mkdir(parents=True)
    target = tmp_path / "epoch0_m1/G0001_Relax-0001"
    target.mkdir()
    with pytest.raises(FileExistsError):
        submission_moves(
            tmp_path,
            {"slurm_batches": [{"upload_directory": str(old / "inputs/batch")}]},
        )
    assert old.is_dir()


def test_completed_structure_pointer_does_not_hide_calculation_json(tmp_path):
    structure = tmp_path / "final.vasp"
    structure.write_text("structure")
    result = tmp_path / "result.json"
    result.write_text("{}")
    task = {
        "result_path": str(structure),
        "calculation_result_path": str(result),
        "stage": "relax_and_feature",
    }
    assert _task_result_path(task, None) == result


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows directory junction")
def test_locked_directory_fails_without_creating_links(tmp_path, monkeypatch):
    project = tmp_path / "locked-project"
    old = project / "submissions/epoch0_m1/Search-group-0001/Relax-0001"
    batch = old / "inputs/batch"
    batch.mkdir(parents=True)
    (batch / "task.json").write_text('{"immutable": true}')
    state_path = project / "workflow_state/state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps({"slurm_batches": [{"upload_directory": str(batch)}]})
    )
    original_rename = Path.rename

    def locked_rename(self, target):
        if self == old:
            raise PermissionError("directory locked")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", locked_rename)
    with pytest.raises(PermissionError, match="directory locked"):
        migrate_submission_layout(project, tmp_path / "backup")
    assert old.is_dir() and not old.is_junction() and not old.is_symlink()
    assert not (project / "submissions/epoch0/G0001_Relax-0001").exists()
    assert (batch / "task.json").read_text() == '{"immutable": true}'
