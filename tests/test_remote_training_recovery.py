import csv
import json
from copy import deepcopy

from phase_agent.tools.local.recover_remote_training import inspect_training_results, recover_remote_training
from phase_agent.runtime.chat_application import _is_status_command
from phase_agent.runtime.status_presentation import format_progress


def returned_job(tmp_path):
    directory = tmp_path / "MLIP-finetune-round-0001"
    root = directory / "results"
    root.mkdir(parents=True)
    job = {"directory": str(directory), "original_model_version": "base",
           "status": "inputs_prepared", "report": {"committees": [{"model_id": "com_1"}]}}
    marker = {"status": "completed", "original_model_version": "base", "committee_count": 1,
              "out_of_fold_structures": 1, "folds": 1, "activated": False}
    model = {"model_id": "com_1", "source_model_version": "base", "is_main_model": True,
             "is_committee_member": True, "training_round": directory.name,
             "remote_model_path": "/hpc/com_1.model", "sha256": "a" * 64}
    (root / "training.finished.json").write_text(json.dumps(marker))
    (root / "models.json").write_text(json.dumps([model]))
    rows = {
        "kfold_metrics.csv": {"fold": "all_out_of_fold", "structures": 1, "force_components": 1,
            "energy_MAE_meV_per_atom": 1, "energy_RMSE_meV_per_atom": 1,
            "force_MAE_meV_per_A": 2, "force_RMSE_meV_per_A": 2},
        "energy_comparison.csv": {"fold": "fold_1", "E_DFT_eV_per_atom": 1, "E_MLIP_eV_per_atom": 1.001},
        "force_comparison.csv": {"fold": "fold_1", "F_DFT_eV_per_A": 1, "F_MLIP_eV_per_A": 1.002}}
    for name, row in rows.items():
        with (root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            if name == "kfold_metrics.csv":
                writer.writerow({**row, "fold": "fold_1"})
    return job, root


def test_read_only_status_and_idempotent_recovery(tmp_path):
    job, root = returned_job(tmp_path)
    state = {"active_model_version": "base", "remote_finetune_jobs": {"j": job},
             "pending_execution_policies": {"old": {"agent_proposal": {"recommended_action": "update_mlip"}}}}
    before = deepcopy(state)
    text = format_progress(state)
    assert "训练结果已回传" in text and "上传完整" not in text
    assert state == before
    recovered, reports = recover_remote_training(state)
    assert reports[0]["status"] == "validation_required"
    assert recovered["remote_finetune_jobs"]["j"]["status"] == "results_received"
    assert "candidate_models" not in recovered and "active_model" not in recovered
    assert recover_remote_training(recovered)[0] == recovered


def test_legacy_relative_paths_are_not_guessed(tmp_path):
    job, root = returned_job(tmp_path)
    models = json.loads((root / "models.json").read_text())
    models[0].pop("remote_model_path")
    models[0].pop("sha256")
    models[0]["model_path"] = "models/com_1.model"
    (root / "models.json").write_text(json.dumps(models))
    report = inspect_training_results(job)
    assert report["status"] == "metadata_required"
    assert len(report["issues"]) == 2


def test_partial_and_wrong_version_do_not_complete(tmp_path):
    job, root = returned_job(tmp_path)
    marker = json.loads((root / "training.finished.json").read_text())
    marker["original_model_version"] = "wrong"
    (root / "training.finished.json").write_text(json.dumps(marker))
    updated, reports = recover_remote_training({"remote_finetune_jobs": {"j": job}})
    assert reports[0]["status"] == "invalid_results"
    assert updated["remote_finetune_jobs"]["j"]["status"] == "inputs_prepared"
    (root / "models.json").unlink()
    assert "models.json" in inspect_training_results(job)["issues"][0]


def test_natural_status_queries():
    for message in ("现在处于什么状态", "现在是什么状态？", "目前处于什么状态"):
        assert _is_status_command(message)
    assert not _is_status_command("现在重新生成训练输入")
