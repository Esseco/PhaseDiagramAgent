import csv
import json
from pathlib import Path

from phase_agent.analysis.feedback.out_data import export_out_data


def _state(tmp_path):
    return {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "phase_diagrams": {
            "mlip": {
                "model_version": "m1",
                "entries": [{
                    "structure_id": "s1", "phase": "O3",
                    "composition": {"Na": 1, "Ni": 1, "O": 2},
                    "ehull": 0.012, "energy_per_atom": -5.4,
                    "structure_path": "relaxed.vasp",
                }],
            }
        },
        "tasks": [{
            "task_id": "t1", "task_key": "relax:1",
            "model_version": "m1", "stage": "relax_and_feature",
            "status": "completed", "structure_id": "s1",
            "branch_id": "b1",
            "outputs": {
                "structure_path": "relaxed.vasp",
                "structure_checksum": "hash1",
                "composition": {"Na": 1, "Ni": 1, "O": 2},
                "actual_phase": "O3", "energy_per_atom": -5.4,
                "qbc": {"energy_per_atom_std": 0.03},
                "features": [1.0, 2.0],
                "extrapolation_monitor": {"final_feature_difference": 0.4},
            },
        }, {
            "task_id": "t2", "task_key": "mc:1",
            "model_version": "m1", "stage": "deep_search",
            "status": "completed", "structure_id": "s1",
            "branch_id": "b1", "outputs": {"structure_path": "relaxed.vasp"},
        }],
        "dft_dataset_records": [], "dft_mlip_comparisons": [],
        "remote_finetune_jobs": {},
    }


def test_export_writes_epoch_and_dft_files(tmp_path):
    state = _state(tmp_path)
    job = tmp_path / "job" / "results"
    job.mkdir(parents=True)
    (job / "kfold_metrics.csv").write_text(
        "fold,structures,force_components,energy_MAE_meV_per_atom,energy_RMSE_meV_per_atom,force_MAE_meV_per_A,force_RMSE_meV_per_A\n"
        "all_out_of_fold,2,6,1,2,3,4\n"
        "current_round_out_of_fold,1,3,5,6,7,8\n", encoding="utf-8"
    )
    state["remote_finetune_jobs"] = {"job1": {
        "directory": str(job.parent), "original_model_version": "m1",
        "status": "results_received", "report": {"epoch": "epoch0"},
    }}
    result = export_out_data(state, tmp_path / "analysis_outputs")
    assert result["status"] == "completed"
    dft = Path(result["dft_error_path"])
    rows = list(csv.DictReader(dft.open(encoding="utf-8")))
    assert {row["metric_scope"] for row in rows} >= {"kfold_all_out_of_fold", "kfold_current_round_out_of_fold"}
    features = Path(result["structure_paths"]["epoch0"])
    feature_rows = list(csv.DictReader(features.open(encoding="utf-8")))
    assert [row["name"] for row in feature_rows] == ["s1", "s1__02"]
    assert feature_rows[0]["m3gnet_features"] == "[1.0,2.0]"
    assert feature_rows[0]["dh_extrapolation"] == "0.4"
    assert feature_rows[0]["na_per_o2"] == "1.0"
    assert feature_rows[0]["ehull_eV_per_atom"] == "0.012"


def test_missing_features_are_explicit(tmp_path):
    state = _state(tmp_path)
    result = export_out_data(state, tmp_path / "analysis_outputs")
    rows = list(csv.DictReader(Path(result["structure_paths"]["epoch0"]).open(encoding="utf-8")))
    missing = next(row for row in rows if row["m3gnet_feature_status"] == "missing")
    assert missing["m3gnet_features"] == ""
    assert result["missing_feature_rows"] == 1


def test_versioned_snapshots_and_latest_are_written(tmp_path):
    state = _state(tmp_path)
    root = tmp_path / "analysis_outputs"
    result = export_out_data(state, root, action="relax_mc", epoch="epoch0")
    assert Path(result["snapshots"]["paths"][0]).is_file()
    assert (root / "out_data" / "versions" / "epoch0" / "relax_mc" / "epoch0_structs_features.csv").is_file()
    assert (root / "out_data" / "latest" / "epoch0_structs_features.csv").is_file()
    assert (root / "latest" / "out_data" / "DFT_err_data.csv").is_file()


def test_dft_snapshot_is_scoped_by_epoch(tmp_path):
    state = _state(tmp_path)
    root = tmp_path / "analysis_outputs"
    export_out_data(state, root, action="dft")
    assert (root / "out_data" / "versions" / "epoch0" / "dft" / "DFT_err_data.csv").is_file()

def test_organization_keeps_out_data_at_root(tmp_path):
    from phase_agent.analysis.feedback.analysis_output_organization import organize_analysis_outputs
    root = tmp_path / "analysis_outputs"
    (root / "out_data").mkdir(parents=True)
    (root / "out_data" / "DFT_err_data.csv").write_text("epoch\n", encoding="utf-8")
    (root / "latest" / "out_data").mkdir(parents=True)
    (root / "latest" / "out_data" / "DFT_err_data.csv").write_text("epoch\n", encoding="utf-8")
    result = organize_analysis_outputs({}, root)
    assert result["status"] in {"completed", "completed_with_legacy", "incomplete"}
    assert (root / "out_data" / "DFT_err_data.csv").is_file()
    assert (root / "latest" / "out_data" / "DFT_err_data.csv").is_file()
