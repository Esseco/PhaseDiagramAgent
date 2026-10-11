import json
from pathlib import Path
import zipfile

import pytest

from phase_agent.tools.remote.transfer_bundle import (
    MANIFEST,
    checksum,
    create_bundle,
    has_registered_return,
    import_bundle,
    import_registered_bundles,
    input_bundle,
    result_bundle,
)


def test_input_archive_keeps_inputs_and_portable_layout_only(tmp_path):
    root = tmp_path / "inputs" / "batch-1"
    (root / "task").mkdir(parents=True)
    for name in ("task/task.json", "task/POSCAR", "GPU.sh", "transfer_bundle.py"):
        (root / name).write_text("input")
    for name in ("task/debug.log", "UPLOAD_AND_SUBMIT.md", "SHA256SUMS"):
        (root / name).write_text("unneeded")
    (root / "results").mkdir()
    (root / "results/result.json").write_text("old result")
    with zipfile.ZipFile(input_bundle(root)) as archive:
        assert set(archive.namelist()) == {
            MANIFEST,
            "batch-1/task/task.json",
            "batch-1/task/POSCAR",
            "batch-1/GPU.sh",
            "batch-1/transfer_bundle.py",
        }
        assert json.loads(archive.read(MANIFEST))["kind"] == "inputs"


def test_result_archive_is_scoped_and_preserves_resume_and_final_structure(tmp_path):
    root = tmp_path / "remote/results"
    task = root / "MC-1"
    task.mkdir(parents=True)
    (root / "MC-other").mkdir()
    (root / "MC-other/result.json").write_text("unrelated")
    payload = {"task_id": "MC-1", "checkpoint": "resume.json.gz"}
    (task / "result.json").write_text(json.dumps(payload))
    for name in ("task.finished.json", "resume.json.gz", "candidate.vasp", "remote_artifacts.json"):
        (task / name).write_bytes(b"required")
    (task / "trace.json.gz").write_bytes(b"large diagnostics")
    archive = result_bundle(root, [task], "batch-1")
    local = tmp_path / "local/results"
    assert import_bundle(archive, local) == 5
    assert (local / "MC-1/resume.json.gz").read_bytes() == b"required"
    assert not (local / "MC-1/trace.json.gz").exists()
    assert not (local / "MC-other").exists()
    assert import_bundle(archive, local) == 5


def test_checksum_failure_and_conflict_publish_nothing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("result.json", "task.finished.json"):
        (source / name).write_text("original")
    archive = Path(
        create_bundle(source, source.iterdir(), tmp_path / "valid.results.zip", kind="results")
    )
    target = tmp_path / "target"
    target.mkdir()
    (target / "result.json").write_text("existing different")
    with pytest.raises(ValueError, match="differs"):
        import_bundle(archive, target)
    assert not (target / "task.finished.json").exists()
    corrupt = tmp_path / "bad.results.zip"
    with zipfile.ZipFile(archive) as old, zipfile.ZipFile(corrupt, "w") as new:
        for name in old.namelist():
            new.writestr(name, b"corrupted" if name == "task.finished.json" else old.read(name))
    with pytest.raises(ValueError, match="checksum"):
        import_bundle(corrupt, tmp_path / "clean")
    assert not (tmp_path / "clean/result.json").exists()


@pytest.mark.parametrize(
    "name", ["../result.json", "/result.json", "C:/result.json", "a/../result.json"]
)
def test_unsafe_archive_paths_are_rejected(tmp_path, name):
    archive = tmp_path / "bad.results.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "data")
        bundle.writestr(MANIFEST, "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        import_bundle(archive, tmp_path / "results")


def test_registered_import_receipt_is_idempotent_and_input_zip_is_not_a_return(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text("{}")
    create_bundle(source, [source / "result.json"], root / "batch.results.zip", kind="results")
    state = {"slurm_batches": [{"results_directory": str(root)}]}
    assert has_registered_return(state)
    assert import_registered_bundles([root]) == []
    assert not has_registered_return(state)
    assert import_registered_bundles([root]) == []
    create_bundle(source, [source / "result.json"], root / "input.results.zip", kind="inputs")
    errors = import_registered_bundles([root])
    assert len(errors) == 1 and "Not a supported" in errors[0]["reason"]


def test_training_bundle_keeps_models_remote(tmp_path):
    root = tmp_path / "remote/results"
    root.mkdir(parents=True)
    for name in (
        "training.finished.json",
        "models.json",
        "kfold_metrics.csv",
        "energy_comparison.csv",
        "force_comparison.csv",
    ):
        (root / name).write_text("result")
    (root / "huge.model").write_bytes(b"model")
    archive = result_bundle(root, [root], "training")
    target = tmp_path / "local"
    assert import_bundle(archive, target) == 5
    assert not (target / "huge.model").exists()
    with pytest.raises(ValueError, match="size limit"):
        import_bundle(archive, tmp_path / "bounded", max_bytes=1)


def test_windows_separator_is_rejected_before_archive_normalization():
    from phase_agent.tools.remote.transfer_bundle import _safe_name

    with pytest.raises(ValueError, match="Unsafe"):
        _safe_name("a\\result.json")


def test_zip_return_reaches_original_scientific_collector(tmp_path):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.remote.export_batch_result import export_batch_result
    import shutil

    batch = tmp_path / "inputs/batch"
    task_root = batch / "RELAX-1"
    task_root.mkdir(parents=True)
    task = {
        "task_id": "RELAX-1",
        "task_key": "key",
        "batch_id": "b1",
        "task_checksum": "input",
        "protocol_version": 1,
        "input_file_version": "task-json-v1",
    }
    (task_root / "task.json").write_text(json.dumps(task))
    final = task_root / "final.vasp"
    final.write_text("final")
    result = {
        **task,
        "stage": "relax_and_feature",
        "status": "completed",
        "outputs": {
            "energy": -1,
            "structure_path": str(final),
            "structure_checksum": checksum(final),
        },
    }
    (task_root / "result.json").write_text(json.dumps(result))
    (task_root / "task.finished.json").write_text(
        json.dumps(
            {**task, "status": "completed", "result_checksum": checksum(task_root / "result.json")}
        )
    )
    remote = tmp_path / "remote/results"
    exported = export_batch_result(task_root, remote)
    archive = result_bundle(remote, [exported], "batch")
    local = tmp_path / "results"
    local.mkdir()
    shutil.copy2(archive, local)
    # No local raw result files; only original input and the downloaded archive.
    (task_root / "result.json").unlink()
    (task_root / "task.finished.json").unlink()
    state = {
        "tasks": [
            {
                **task,
                "stage": "relax_and_feature",
                "status": "pending",
                "input_path": str(task_root / "task.json"),
                "result_path": str(local / exported.name / "result.json"),
            }
        ],
        "slurm_batches": [
            {"batch_id": "b1", "upload_directory": str(batch), "results_directory": str(local)}
        ],
    }
    runner = RemoteBatchRunner(tmp_path / "unused", worker_command=[])
    collected = runner.collect_results_with_report(state)
    assert collected["report"]["recovered_count"] == 1
    assert collected["report"]["transfer_errors"] == []
    assert collected["results"][0]["outputs"]["structure_path"] == str(
        (local / exported.name / "final.vasp").resolve()
    )
    # Passing transfer hashes never bypasses original scientific input identity.
    state["tasks"][0]["task_checksum"] = "different"
    rejected = runner.collect_results_with_report(state)
    assert rejected["report"]["recovered_count"] == 0
    assert rejected["report"]["invalid_count"] == 1


def test_existing_input_archive_keeps_frozen_inputs_unchanged(tmp_path):
    from phase_agent.tools.remote.manual_upload_runner import package_existing_manual_inputs

    root = tmp_path / "inputs/batch-1"
    (root / "task").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps([{"stage": "deep_search", "input_path": "task/task.json"}])
    )
    (root / "task/task.json").write_text('{"task_checksum":"original"}')
    (root / "run_mlip_batch.py").write_text(
        "from phase_agent.tools.remote.run_mlip_batch import main\n"
    )
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    archive = package_existing_manual_inputs(root)
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("batch-1/task/task.json") == before["task/task.json"]
        assert b"from _handoff_mlip_batch import main" in bundle.read("batch-1/run_mlip_batch.py")
        assert b"from transfer_bundle import result_bundle" in bundle.read(
            "batch-1/_handoff_mlip_batch.py"
        )
        assert "batch-1/transfer_bundle.py" in bundle.namelist()


def test_training_archive_recovery_is_owned_by_collector_not_status(tmp_path):
    from tests.test_remote_training_recovery import returned_job
    from phase_agent.tools.local.recover_remote_training import (
        inspect_training_results,
        recover_remote_training,
    )

    job, root = returned_job(tmp_path)
    archive = result_bundle(root, [root], "training")
    for path in root.iterdir():
        if path.is_file() and str(path) != archive:
            path.unlink()
    assert inspect_training_results(job) is None
    state, reports = recover_remote_training({"remote_finetune_jobs": {"j": job}})
    assert reports[0]["status"] == "validation_required"
    assert state["remote_finetune_jobs"]["j"]["status"] == "results_received"
    assert not state["remote_finetune_jobs"]["j"].get("activated")
    assert "active_model" not in state


def test_ssh_results_directory_avoids_logs_and_duplicate_archives(tmp_path):
    import subprocess
    from phase_agent.tools.remote.ssh_adapter import OpenSSHTransport

    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ssh":
            command = argv[-1]
            if command.startswith("cat "):
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
            if command.startswith("find "):
                names = [
                    "task/result.json",
                    "task/task.finished.json",
                    "task/final.vasp",
                    "task/checkpoint.json.gz",
                    "task/trace.json.gz",
                    "task/task.stdout.log",
                    "batch.results.zip",
                ]
                return subprocess.CompletedProcess(
                    argv, 0, stdout="\n".join("/remote/results/" + n for n in names), stderr=""
                )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    report = OpenSSHTransport("HPC", runner=runner).download_tree("/remote/results", tmp_path)
    assert set(report["files"]) == {
        "task/result.json",
        "task/task.finished.json",
        "task/final.vasp",
        "task/checkpoint.json.gz",
    }
    assert not any("-r" in call for call in calls)
