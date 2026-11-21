import hashlib
import json
from phase_agent.analysis.feedback.training_parity import publish_training_parity
from phase_agent.tools.local.recover_remote_training import FILES


def test_training_parity_uses_bound_results_and_detects_changes(tmp_path):
    root = tmp_path / "analysis_outputs"
    root.mkdir()
    result = tmp_path / "job" / "results"
    result.mkdir(parents=True)
    files = {
        "training.finished.json": json.dumps(
            {
                "status": "completed",
                "original_model_version": "old",
                "out_of_fold_structures": 1,
                "folds": 1,
            }
        ),
        "models.json": "[]",
        "kfold_metrics.csv": "fold,structures,force_components\nall_out_of_fold,1,1\n",
        "energy_comparison.csv": "fold,task_id,E_DFT_eV_per_atom,E_MLIP_eV_per_atom\nf1,t1,-1,-1.1\n",
        "force_comparison.csv": "fold,task_id,F_DFT_eV_per_A,F_MLIP_eV_per_A\nf1,t1,0.1,0.2\n",
    }
    h = hashlib.sha256()
    for name in FILES:
        (result / name).write_text(files[name], encoding="utf8")
        h.update(name.encode())
        h.update((result / name).read_bytes())
    fp = h.hexdigest()
    version = "remote-j-" + fp[:12]
    job = {
        "directory": str(result.parent),
        "original_model_version": "old",
        "activated": True,
        "returned_results": {"fingerprint": fp, "status": "validation_required"},
        "training_handoff": {"candidate_model_version": version},
    }
    state = {
        "active_model_version": version,
        "upload_layout": {"model_rounds": {"old": 1}},
        "remote_finetune_jobs": {"j": job},
    }
    report = publish_training_parity(state, root)
    assert report["latest"]["status"] == "completed"
    assert (root / "epoch0/KFold_Error/job/energy_parity.png").is_file()
    assert (root / "latest/KFold_Error/force_parity.png").is_file()
    (result / "energy_comparison.csv").write_text("modified", encoding="utf8")
    report = publish_training_parity(state, root)
    assert report["latest"]["status"] == "unavailable"
    assert "指纹" in report["jobs"][0]["reason"]


def test_kfold_excludes_only_proven_failed_tasks_without_changing_training_evidence(
    tmp_path, monkeypatch
):
    from copy import deepcopy
    from phase_agent.analysis.feedback import training_parity
    import phase_agent.analysis.feedback.dft_parity_plots as plots

    root = tmp_path / "analysis_outputs"
    root.mkdir()
    rows = [{"task_id": tid, "comparison_status": "completed"} for tid in ("bad", "good")]
    seen = []
    monkeypatch.setattr(
        training_parity,
        "_verified",
        lambda job, key: ("f" * 64, "m", deepcopy(rows), deepcopy(rows)),
    )
    monkeypatch.setattr(
        plots,
        "export_parity_plots",
        lambda e, f, d, l: (
            seen.append((deepcopy(e), deepcopy(f))) or {"status": "completed", "metrics": {}}
        ),
    )
    state = {
        "active_model_version": "m",
        "upload_layout": {"model_rounds": {"old": 1}},
        "remote_finetune_jobs": {
            "j": {
                "directory": str(tmp_path / "job"),
                "original_model_version": "old",
                "activated": True,
            }
        },
        "tasks": [
            {
                "task_id": "r",
                "model_version": "m0",
                "outputs": {
                    "structure_checksum": "sha",
                    "extrapolation_monitor": {"extrapolation_failure": True},
                },
            },
            {
                "task_id": "bad",
                "stage": "dft_single_point",
                "model_version": "m0",
                "input_structure_sha256": "sha",
            },
            {
                "task_id": "good",
                "stage": "dft_single_point",
                "model_version": "m0",
                "input_structure_sha256": "other",
            },
        ],
    }
    before = deepcopy(state)
    report = training_parity.publish_training_parity(state, root)
    assert report["jobs"][0]["excluded_extrapolation_failed_task_ids"] == ["bad"]
    assert all(
        [r["task_id"] for r in e] == ["good"] and [r["task_id"] for r in f] == ["good"]
        for e, f in seen
    )
    assert state == before
