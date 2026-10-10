from pathlib import Path
from phase_agent.analysis.phase.phase_snapshot_paths import phase_snapshot_directory
from phase_agent.analysis.feedback.export_dft_products import export_dft_products
from tests.test_dft_comparison_csv import add_result


def test_shared_epoch_paths(tmp_path):
    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1, "m2": 2}}}
    for method in ("mlip", "dft", "combined"):
        path = phase_snapshot_directory(tmp_path, method, "m1", state=state)
        assert path == tmp_path / "epoch0_m1" / "phase_diagrams" / method / "history"
    assert "epoch1_m2" in phase_snapshot_directory(tmp_path, "mlip", "m2", state=state).parts


def test_phase_csv_epoch_and_unchanged_reuse(tmp_path):
    import csv
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from tests.test_na_eform_phase_csv import record
    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1}}}
    records = [record("left", 0, -3), record("right", 1, -4)]
    first = update_phase_diagram(records, active_model_version="m1", output_directory=tmp_path,
                                output_state=state)["diagrams"]["mlip"]
    path = Path(first["csv_path"])
    assert path == tmp_path / "epoch0_m1/phase_diagrams/mlip/phase_diagram.csv"
    with path.open(encoding="utf-8-sig") as stream:
        assert list(csv.DictReader(stream))[0]["epoch"] == "epoch0"
    stamp = path.stat().st_mtime_ns
    repeated = update_phase_diagram(records, active_model_version="m1", output_directory=tmp_path,
                                   output_state=state)["diagrams"]["mlip"]
    assert repeated["csv_path"] == first["csv_path"]
    assert path.stat().st_mtime_ns == stamp


def test_comparison_csv_has_epoch_and_round(tmp_path):
    import csv
    state = {}
    add_result(state)
    export_dft_products(state, tmp_path)
    product = next(iter(state["dft_result_exports"].values()))
    assert "epoch0_m1" in Path(product["directory"]).parts
    for key in ("energy_csv_path", "force_csv_path", "metrics_csv_path"):
        with open(product[key], encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and rows[0]["epoch"] == "epoch0"
        assert rows[0]["dft_round"].startswith("DFT-round-0001_")
    assert Path(state["output_index_path"]).suffix == ".md"
    stamp = Path(state["output_index_path"]).stat().st_mtime_ns
    export_dft_products(state, tmp_path)
    assert Path(state["output_index_path"]).stat().st_mtime_ns == stamp


def test_missing_training_directory_blocks_without_generation(tmp_path):
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune
    from phase_agent.configuration.defaults.default_mace_committee_config import default_mace_committee_config
    state = {}
    for i in range(10):
        add_result(state, task_id=f"T{i}")
    state["remote_finetune_jobs"] = {"old": {"directory": str(tmp_path / "missing"),
        "original_model_version": "m1", "status": "inputs_prepared"}}
    settings = default_mace_committee_config()
    settings["labels"]["include_stress"] = False
    config = {"mlip_finetune": settings, "upload_batches_directory": str(tmp_path),
        "mlip": {"version": "m1", "model_path": "/remote/m1.model"},
        "python_environments": {"remote_mlip": "mace"}}
    result = prepare_remote_finetune(state, config)
    assert result["status"] == "confirmation_required"
    assert not list(tmp_path.iterdir())


def test_changed_training_plan_does_not_create_next_round(tmp_path):
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune
    from phase_agent.configuration.defaults.default_mace_committee_config import default_mace_committee_config
    state = {}
    for i in range(10):
        add_result(state, task_id=f"T{i}")
    settings = default_mace_committee_config()
    settings["labels"]["include_stress"] = False
    config = {"mlip_finetune": settings, "upload_batches_directory": str(tmp_path),
        "mlip": {"version": "m1", "model_path": "/remote/m1.model"},
        "python_environments": {"remote_mlip": "mace"}}
    prepared = prepare_remote_finetune(state, config)
    job = next(iter(prepared["state"]["remote_finetune_jobs"].values()))
    path = Path(job["directory"]) / "inputs/com_1/mace_train.yaml"
    original = path.read_bytes()
    settings["training"]["patience"] = 19
    blocked = prepare_remote_finetune(prepared["state"], config)
    assert blocked["status"] == "confirmation_required"
    assert path.read_bytes() == original
    assert not (Path(job["directory"]).parent / "MLIP-finetune-round-0002").exists()
    # A missing unrelated historical folder must not block reuse of this
    # exact registered dataset/settings combination.
    settings["training"]["patience"] = 20
    prepared["state"]["remote_finetune_jobs"]["historic"] = {
        "directory": str(tmp_path / "deleted-historic-inputs"),
        "original_model_version": "m1", "status": "inputs_prepared"}
    reused = prepare_remote_finetune(prepared["state"], config)
    assert reused["status"] == "awaiting_remote_training"
    assert Path(job["directory"]).is_dir()
