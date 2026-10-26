"""Saved analysis navigation stays separate from raw submission records."""

from copy import deepcopy
from pathlib import Path

from phase_agent.analysis.feedback.output_catalog import catalog_markdown, output_catalog


def test_raw_submission_kinds_remain_machine_readable_but_human_index_links_once(tmp_path):
    output = tmp_path / "analysis_outputs"
    root = tmp_path / "custom-submissions"
    batch = root / "epoch0/G0001_DFT-0001/inputs/batch"
    batch.mkdir(parents=True)
    training = root / "epoch0/MLIP-finetune-round-0001"
    training.mkdir()
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "slurm_batches": [{"upload_directory": str(batch), "model_version": "m1"}],
        "remote_finetune_jobs": {
            "train1": {
                "directory": str(training),
                "original_model_version": "m1",
                "status": "inputs_prepared",
            }
        },
    }
    before = deepcopy(state)
    rows = output_catalog(state)
    assert {row["product"] for row in rows} == {
        "submission_files",
        "returned_results",
        "remote_training_inputs",
        "training_returned_results",
    }
    text = catalog_markdown(rows, output)
    assert "../custom-submissions/README.md" in text
    assert text.count("../custom-submissions/README.md") == 1
    for row in rows:
        assert row["product"] not in text
    assert "G0001_DFT-0001" not in text and "MLIP-finetune-round-0001" not in text
    assert state == before


def test_registered_round_reports_and_plots_are_indexed_with_current_model_first(tmp_path):
    output = tmp_path / "analysis_outputs"
    files = {}
    for epoch, label in (("epoch10", "old"), ("epoch2", "current")):
        folder = output / epoch / "phase_diagrams/mlip/quality/rounds"
        folder.mkdir(parents=True)
        report, plot = folder / (label + ".md"), folder / (label + ".png")
        report.write_text("saved report", encoding="utf-8")
        plot.write_bytes(b"saved plot")
        files[label] = (report, plot)
    state = {
        "active_model_version": "m2",
        "upload_layout": {"model_rounds": {"m1": 11, "m2": 3}},
        "relax_mc_batches": {
            label: {
                "model_version": model,
                "package_label": label,
                "phase_analysis": {
                    "status": "completed",
                    "markdown_path": str(files[label][0]),
                    "plot_paths": [str(files[label][1])],
                },
            }
            for label, model in (("old", "m1"), ("current", "m2"))
        },
    }
    before = deepcopy(state)
    rows = output_catalog(state)
    assert len(rows) == 4
    assert {row["product"] for row in rows} == {"relax_mc_report", "relax_mc_plot"}
    assert all(row["exists"] for row in rows)
    text = catalog_markdown(rows, output)
    current, history = text.split("<details>", 1)
    assert "epoch2 · m2" in current and "current.md" in current and "current.png" in current
    assert "epoch10 · m1" not in current
    assert "old.md" in history and "old.png" in history
    assert "历史模型分析" in history
    assert state == before


def test_missing_registered_round_artifact_is_explicit_without_scientific_recalculation(tmp_path):
    report = tmp_path / "epoch0/phase_diagrams/mlip/quality/rounds/missing.md"
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "relax_mc_batches": {
            "b1": {
                "model_version": "m1",
                "phase_analysis": {"markdown_path": str(report), "status": "completed"},
            }
        },
    }
    rows = output_catalog(state)
    assert len(rows) == 1 and rows[0]["exists"] is False
    text = catalog_markdown(rows, tmp_path)
    assert "missing.md" in text and "缺失" in text
    assert not report.exists()
