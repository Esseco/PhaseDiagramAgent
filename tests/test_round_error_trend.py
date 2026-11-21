from phase_agent.analysis.feedback.round_error_trend import publish_round_error_trend


def test_trend_only_uses_verified_selected_or_parent_first_pairs(tmp_path):
    root = tmp_path / "analysis_outputs"
    (root / "epoch0").mkdir(parents=True)
    (root / "epoch1").mkdir()

    def row(tid, key, model="m0"):
        return {
            "task_id": tid,
            "task_key": key,
            "structure_id": "S-" + tid,
            "model_version": model,
            "status": "completed",
            "converged": True,
            "checks_passed": True,
            "training_ready": True,
            "final_frame_index": 0,
            "training_frame_index": 0,
            "atom_count": 1,
            "energy": -1.0,
            "energy_unit": "eV",
            "forces": [[0.1, 0.2, 0.3]],
            "forces_unit": "eV/angstrom",
            "mlip_prediction": {
                "model_version": model,
                "energy": -0.98,
                "energy_unit": "eV",
                "forces": [[0.2, 0.3, 0.4]],
                "forces_unit": "eV/angstrom",
            },
        }

    records = [
        row("q", "q-key"),
        row("p", "p-key"),
        row("unverified", "u-key"),
        row("new", "new-key", "m1"),
    ]
    tasks = [
        {
            "task_id": r["task_id"],
            "task_key": r["task_key"],
            "model_version": r["model_version"],
            "object_id": "DIRECT-q" if r["task_id"] == "q" else r["structure_id"],
        }
        for r in records
    ]
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m0": 1, "m1": 2}},
        "tasks": tasks,
        "dft_dataset_records": records,
        "dft_mlip_comparisons": [{"task_id": r["task_id"], "status": "completed"} for r in records],
        "qbc_selection_history": [
            {"accepted": [{"task_key": "q-key", "candidate_id": "DIRECT-q"}]}
        ],
        "parent_first_recovery_batches": {
            "b": {"sampling_result_sha256": "a" * 64, "model_version": "m0", "dft_task_ids": ["p"]}
        },
    }
    report = publish_round_error_trend(state, root)
    assert [r["status"] for r in report["rows"]] == ["completed", "unavailable"]
    r = report["rows"][0]
    assert r["structures"] == 2 and r["selection_unverified"] == 1 and r["force_components"] == 6
    assert abs(r["energy_mae_eV_per_atom"] - 0.02) < 1e-8
    state["tasks"][0].update(stage="dft_single_point", input_structure_sha256="failed-sha")
    state["tasks"].append(
        {
            "task_id": "source",
            "model_version": "m0",
            "outputs": {
                "structure_checksum": "failed-sha",
                "extrapolation_monitor": {"extrapolation_failure": True},
            },
        }
    )
    filtered = publish_round_error_trend(state, root)["rows"][0]
    assert filtered["structures"] == 1
    assert filtered["excluded_extrapolation_failed"] == 1
    assert filtered["excluded_failed_previously_paired"] == 1
    assert filtered["selection_unverified"] == 1
    for name in (
        "Round_Error_E_MAE_by_epoch.png",
        "Round_Error_F_MAE_by_epoch.png",
        "Round_Error_MAE_by_epoch.csv",
    ):
        assert (root / "latest" / name).stat().st_size > 0


def test_trend_outputs_only_png(tmp_path):
    root = tmp_path / "analysis_outputs"
    (root / "epoch0").mkdir(parents=True)
    publish_round_error_trend({}, root)
    assert len(list((root / "latest").glob("*.png"))) == 2
    assert not list((root / "latest").glob("*.pdf"))
