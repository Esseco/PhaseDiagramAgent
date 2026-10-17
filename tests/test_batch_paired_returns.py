import json
import shutil
import zipfile
from pathlib import Path

import pytest

from tests.test_round_transfer import setup_round, completed
from phase_agent.tools.remote.submission_handoff import ensure_waiting_handoffs
from phase_agent.tools.remote.transfer_bundle import (
    publish_return,
    registered_result_directories,
    registered_return_locations,
    import_registered_bundles,
    has_registered_return,
)


def test_new_batches_roundtrip_to_paired_output_and_registered_original_paths(tmp_path):
    local, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    handoff = state["slurm_batches"][0]["handoff"]
    upload = Path(handoff["folder"])
    output = Path(handoff["return_directory"])
    assert upload.name.endswith("_Batch01_input")
    assert output == upload.with_name(upload.name[:-6] + "_output") and output.is_dir()
    remote = tmp_path / "hpc" / upload.name
    remote.mkdir(parents=True)
    with zipfile.ZipFile(handoff["upload_archive"]) as bundle:
        bundle.extractall(remote)
    shutil.copy2(handoff["upload_archive"], remote / "inputs.zip")
    completed(remote / "SP/results", "one")
    archive = Path(publish_return(remote, "SP/results"))
    assert archive.parent == remote.with_name(output.name)
    shutil.copy2(archive, output / archive.name)
    assert has_registered_return(state)
    assert (
        import_registered_bundles(
            registered_result_directories(state), locations=registered_return_locations(state)
        )
        == []
    )
    assert (local / "SP/results/one/result.json").is_file()
    assert not has_registered_return(state)
    assert (
        import_registered_bundles(
            registered_result_directories(state), locations=registered_return_locations(state)
        )
        == []
    )


def test_registered_output_mapping_rejects_conflicting_directory(tmp_path):
    _, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    state["slurm_batches"][0]["handoff"]["return_directory"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="batch_output_directory_mismatch"):
        registered_return_locations(state)


def test_payload_substitution_keeps_manifest_and_original_checkpoint(tmp_path):
    from phase_agent.tools.remote.transfer_bundle import create_bundle, validate_bundle

    source = tmp_path / "workflow_state.json"
    source.write_text('{"large_history":true}')
    archive = create_bundle(
        tmp_path,
        [source],
        tmp_path / "results.zip",
        kind="results",
        payloads={"workflow_state.json": b'{"tasks":[]}'},
    )
    validate_bundle(archive)
    with zipfile.ZipFile(archive) as bundle:
        assert json.loads(bundle.read("workflow_state.json")) == {"tasks": []}
    assert json.loads(source.read_text()) == {"large_history": True}


def test_scientific_closure_drops_duplicate_workdir_and_unreferenced_trace(tmp_path):
    from phase_agent.tools.relax_mc_batch.return_payload import result_files, compact_state

    result = tmp_path / "jobs/results/task"
    result.mkdir(parents=True)
    (result / "final.vasp").write_text("geometry")
    (result / "result.json").write_text(json.dumps({"outputs": {"structure_path": "final.vasp"}}))
    (result / "task.finished.json").write_text("{}")
    (result / "mc_occupations.bin").write_bytes(b"occupancy")
    (result / "trace.json.gz").write_bytes(b"not needed")
    duplicate = tmp_path / "jobs/inputs/task"
    shutil.copytree(result, duplicate)
    task = {
        "task_id": "t",
        "status": "completed",
        "result_path": str(result / "result.json"),
        "outputs": {"structure_path": str(result / "final.vasp")},
    }
    state = {"tasks": [task], "irrelevant_history": ["large"]}
    files = result_files(tmp_path, state, {"branch_candidates": [], "mc_pool": []})
    assert {p.name for p in files} == {
        "result.json",
        "task.finished.json",
        "final.vasp",
        "mc_occupations.bin",
    }
    assert not any(duplicate in p.parents for p in files)
    assert "irrelevant_history" not in compact_state(state)


@pytest.mark.parametrize("project", ["O3-1ele-NaCoO2", "O3-1ele-NaCrO2", "O3-2ele-NaFeMn"])
def test_project_new_package_to_verified_collector(tmp_path, project):
    from phase_agent.tools.remote.submission_handoff import prepare_handoff
    from phase_agent.tools.remote.export_batch_result import export_batch_result
    from phase_agent.tools.remote.integrity import file_checksum
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner

    local = tmp_path / project / "submissions/epoch0_m1"
    worker = local / "SP/inputs/batch/task"
    worker.mkdir(parents=True)
    identity = dict(
        task_id="T1",
        task_key="key",
        batch_id="batch",
        config_version="v1",
        model_version="m1",
        task_checksum="input-checksum",
        protocol_version=1,
        input_file_version="task-json-v1",
    )
    (worker / "task.json").write_text(json.dumps(identity))
    (worker / "GPU.sh").write_text("#!/bin/bash\ntrue\n")
    h = prepare_handoff(local, [worker / "GPU.sh"])
    remote = tmp_path / "hpc" / Path(h["folder"]).name
    remote.mkdir(parents=True)
    shutil.copy2(h["upload_archive"], remote / "inputs.zip")
    with zipfile.ZipFile(h["upload_archive"]) as z:
        z.extractall(remote)
    task = remote / "SP/inputs/batch/task"
    (task / "evaluated.vasp").write_text("fake geometry")
    payload = {
        **identity,
        "status": "completed",
        "stage": "relax_and_feature",
        "outputs": {
            "structure_path": "evaluated.vasp",
            "structure_checksum": file_checksum(task / "evaluated.vasp"),
            "energy": -12.5,
            "energy_unit": "eV",
            "composition": {"Na": 1, "Cr": 1, "O": 2},
            "relax_stopped_normally": True,
        },
    }
    (task / "result.json").write_text(json.dumps(payload))
    (task / "task.finished.json").write_text(
        json.dumps(
            {
                **identity,
                "status": "completed",
                "result_checksum": file_checksum(task / "result.json"),
            }
        )
    )
    export_batch_result(task, remote / "SP/results", create_archive=False)
    returned = Path(publish_return(remote, "SP/results"))
    shutil.copy2(returned, Path(h["return_directory"]) / returned.name)
    state = {
        "tasks": [
            {
                **identity,
                "stage": "relax_and_feature",
                "status": "pending",
                "input_path": str(worker / "task.json"),
                "result_path": str(local / "SP/results/task/result.json"),
            }
        ],
        "slurm_batches": [
            {
                "batch_id": "batch",
                "upload_directory": str(worker.parent),
                "results_directory": str(local / "SP/results"),
                "task_ids": ["T1"],
                "handoff": h,
            }
        ],
    }
    recovered = RemoteBatchRunner(
        tmp_path / "unused", worker_command=[]
    ).collect_results_with_report(state)
    assert recovered["report"]["transfer_errors"] == []
    assert recovered["report"]["recovered_count"] == 1, recovered
    assert recovered["results"][0]["outputs"]["energy"] == -12.5
    assert Path(recovered["results"][0]["outputs"]["structure_path"]).is_file()


def test_wrong_batch_archive_rejected_before_import(tmp_path):
    local, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    h = state["slurm_batches"][0]["handoff"]
    from phase_agent.tools.remote.transfer_bundle import create_bundle

    fake = tmp_path / "wrong"
    fake.mkdir()
    (fake / "result.json").write_text("{}")
    archive = Path(h["return_directory"]) / "wrong.results.zip"
    create_bundle(
        fake,
        [fake / "result.json"],
        archive,
        kind="results",
        binding={"batch": "other", "input_sha256": "wrong"},
    )
    errors = import_registered_bundles(
        registered_result_directories(state), locations=registered_return_locations(state)
    )
    assert errors and "returned_package_binding_mismatch" in errors[0]["reason"]
    assert not (local / "result.json").exists()


def test_all_referenced_pool_structures_returned_without_whole_worker_tree(tmp_path):
    from phase_agent.tools.relax_mc_batch.return_payload import result_files

    out = tmp_path / "jobs/results/t"
    out.mkdir(parents=True)
    work = tmp_path / "jobs/inputs/b/t"
    (work / "pool").mkdir(parents=True)
    (out / "result.json").write_text("{}")
    (out / "task.finished.json").write_text("{}")
    (out / "final.vasp").write_text("final")
    (out / "mc_occupations.bin").write_bytes(b"history")
    (work / "pool/rank_1.vasp").write_text("candidate")
    (work / "pool/unreferenced.vasp").write_text("old")
    task = {
        "task_id": "t",
        "status": "completed",
        "stage": "deep_search",
        "input_path": str(work / "task.json"),
        "result_path": str(out / "final.vasp"),
        "outputs": {
            "structure_path": str(out / "final.vasp"),
            "pool_summary": [{"structure_file": "rank_1.vasp"}],
        },
    }
    candidates = {"mc_pool": [{"source_task_id": "t", "structure_file": "rank_1.vasp"}]}
    selected = result_files(tmp_path, {"tasks": [task]}, candidates)
    assert work / "pool/rank_1.vasp" in selected
    assert work / "pool/unreferenced.vasp" not in selected
    assert candidates["mc_pool"][0]["structure_path"] == str(work / "pool/rank_1.vasp")


def test_training_manifest_followup_roundtrip_without_mutating_original(tmp_path, monkeypatch):
    import subprocess
    import sys
    from tests.test_training_handoff import setup
    from phase_agent.tools.local.training_handoff import advance_training_handoffs
    from phase_agent.tools.local.recover_remote_training import recover_remote_training

    state, results, inputs = setup(tmp_path)
    remote_training = tmp_path / "hpc-training/inputs"
    (remote_training / "com_1").mkdir(parents=True)
    (remote_training / "com_1/com_1.model").write_bytes(b"fake trained model; no model load")
    plan = {
        "committees": [{"model_id": "com_1"}],
        "main_model": {"final_model": "com_1"},
        "original_model_version": "base",
        "training_round": results.parent.name,
    }
    (inputs / "training_plan.json").write_text(json.dumps(plan))
    # Emulate the HPC path lookup on Windows; no model evaluation or training.
    monkeypatch.setattr(
        "phase_agent.tools.local.paired_training.remote_training_inputs",
        lambda *a: remote_training.as_posix(),
    )
    (results / "remote_context.json").write_text(
        json.dumps({"input_directory": remote_training.as_posix()})
    )
    models = json.loads((results / "models.json").read_text())
    models[0]["sha256"] = ""
    (results / "models.json").write_text(json.dumps(models))
    original = (results / "models.json").read_bytes()
    current, waits = advance_training_handoffs(state, {})
    h = waits[0]["handoff"]
    assert {p.name for p in Path(h["folder"]).iterdir()} == {
        "inputs.zip",
        "dispatch_tasks.py",
        "submit_all.sh",
    }
    remote = tmp_path / "cluster" / Path(h["folder"]).name
    remote.mkdir(parents=True)
    shutil.copy2(h["upload_archive"], remote / "inputs.zip")
    with zipfile.ZipFile(h["upload_archive"]) as z:
        assert not any("train.xyz" in n for n in z.namelist())
        z.extractall(remote)
    ran = subprocess.run(
        [sys.executable, str(remote / "inputs/manifest_job.py")], capture_output=True, text=True
    )
    assert ran.returncode == 0, ran.stderr
    # The portable collector is executed on Windows here; emulate its Linux
    # absolute paths in the fake cluster's output before transferring.
    manifest = next((remote / "results").glob("models-*.json"))
    rows = json.loads(manifest.read_text())
    for row in rows:
        row["model_path"] = row["remote_model_path"] = "/hpc/inputs/com_1/com_1.model"
    manifest.write_text(json.dumps(rows))
    archive = Path(publish_return(remote, "results"))
    shutil.copy2(archive, Path(h["return_directory"]) / archive.name)
    recovered, reports = recover_remote_training(current)
    assert reports[0]["status"] == "validation_required", reports
    assert (results / "models.json").read_bytes() == original
    assert reports[0]["models"][0]["sha256"]
    assert recovered["active_model_version"] == "base"


def test_missing_legacy_training_remote_location_is_explicit(tmp_path):
    from tests.test_training_handoff import setup
    from phase_agent.tools.local.training_handoff import advance_training_handoffs

    state, results, inputs = setup(tmp_path)
    (results / "remote_context.json").unlink()
    models = json.loads((results / "models.json").read_text())
    models[0]["sha256"] = ""
    (results / "models.json").write_text(json.dumps(models))
    _, waits = advance_training_handoffs(state, {})
    assert waits[0]["script"] is None and "remote_training_inputs_directory" in waits[0]["reason"]
    assert "handoff" not in waits[0]


def test_validation_followup_return_is_imported_and_approved_separately(tmp_path):
    from tests.test_training_handoff import setup, config
    from phase_agent.tools.local.training_handoff import advance_training_handoffs

    state, results, inputs = setup(tmp_path)
    settings = config(tmp_path)
    current, waits = advance_training_handoffs(state, settings)
    h = waits[0]["handoff"]
    remote = tmp_path / "cluster" / Path(h["folder"]).name
    remote.mkdir(parents=True)
    shutil.copy2(h["upload_archive"], remote / "inputs.zip")
    with zipfile.ZipFile(h["upload_archive"]) as z:
        z.extractall(remote)
        assert "inputs/_shared_data/train.xyz" in z.namelist()
        assert "inputs/run_training.sh" not in z.namelist()
    plan = json.loads((remote / "inputs/validation_request.json").read_text())
    metrics = {
        "energy_mae": 0.003,
        "force_rmse": 0.06,
        "critical_failure_fraction": 0,
        "near_hull_ranking_reversals": 0,
    }
    output = remote / "results"
    output.mkdir()
    (output / ("validation-" + plan["request_id"] + ".json")).write_text(
        json.dumps(
            {
                "request_id": plan["request_id"],
                "data_sha256": plan["data_sha256"],
                "status": "completed",
                "structures": 2,
                "metrics": {"old_model": metrics, "new_model": metrics},
            }
        )
    )
    archive = Path(publish_return(remote, "results"))
    shutil.copy2(archive, Path(h["return_directory"]) / archive.name)
    from phase_agent.tools.local.recover_remote_training import recover_remote_training

    current, reports = recover_remote_training(current)
    assert reports[0]["status"] == "validation_required", reports
    recovered, waits = advance_training_handoffs(current, settings)
    assert waits[0]["stage"] == "awaiting_activation_approval"
    assert recovered["active_model_version"] == "base"


def test_worker_checkpoint_is_retained_for_existing_resume_contract(tmp_path):
    from phase_agent.tools.relax_mc_batch.return_payload import result_files

    worker = tmp_path / "jobs/inputs/t"
    worker.mkdir(parents=True)
    checkpoint = worker / "status.json.gz"
    checkpoint.write_bytes(b"resume")
    directory = tmp_path / "jobs/results/t"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))
    (directory / "task.finished.json").write_text("{}")
    task = {
        "task_id": "t",
        "status": "completed",
        "input_path": str(worker / "task.json"),
        "result_path": str(directory / "result.json"),
    }
    files = result_files(tmp_path, {"tasks": [task]}, {})
    assert checkpoint in files


def test_remote_diagnostic_is_visible_without_results(tmp_path):
    _, state = setup_round(tmp_path)
    ensure_waiting_handoffs(state)
    h = state["slurm_batches"][0]["handoff"]
    with zipfile.ZipFile(Path(h["return_directory"]) / "diagnostics.zip", "w") as z:
        z.writestr("dispatch-error.json", json.dumps({"error": "missing HPC model"}))
    assert has_registered_return(state)
    errors = import_registered_bundles([], locations=registered_return_locations(state))
    assert "missing HPC model" in errors[0]["reason"]


def test_clock_and_publish_exit_hooks_are_composed_and_helpers_packed(tmp_path):
    from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script
    from phase_agent.tools.remote.submission_handoff import prepare_handoff
    root=tmp_path/'epoch0_m1';inputs=root/'inputs';inputs.mkdir(parents=True)
    script=inputs/'GPU.sh'
    write_unix_shell_script(script,'#!/bin/bash\n#SBATCH --gres=gpu:1\nset -euo pipefail\ntrue\n')
    h=prepare_handoff(root,[script],payload_files=[script])
    with zipfile.ZipFile(h['upload_archive']) as z:
        assert 'inputs/_gpu_accounting.py' in z.namelist()
        assert 'inputs/_gpu_transfer_bundle.py' in z.namelist()
        text=z.read('inputs/GPU.sh').decode()
    assert text.index('trap _gpu_usage_finish EXIT') < text.index('phase_status=$?') < text.index('set -euo pipefail')
    final_trap=next(line for line in text.splitlines() if 'phase_status=$?' in line)
    assert final_trap.index('_gpu_accounting.py') < final_trap.index('_round_transfer.py')
    assert 'trap - EXIT' in final_trap and 'exit "$phase_status"' in final_trap
