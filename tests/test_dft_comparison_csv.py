"""Plot-ready exports use saved same-frame pairs, explicit coverage and lineage."""
import csv
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
import pytest

from analysis_layer.feedback.dft_result_products import record_dft_products, export_dft_products
from analysis_layer.feedback.export_dft_products import dft_product_path


def add_result(state, *, task_id="T1", version="m1", operation="abcdef", round_index=1,
               group=1, atoms=2, energy_error=1., force_error=.2, evaluator=True, parent=None):
    species = ["Na", *(["Fe"] * (atoms-2)), "O"]
    structure = Structure(Lattice.cubic(5), species, [[i/atoms, 0, 0] for i in range(atoms)])
    dft_forces = np.asarray([[.1, -.2, .3]] * atoms)
    task = {"task_id": task_id, "model_version": version, "stage": "dft_single_point",
            "upload_operation_id": operation, "search_group_index": group,
            "parent_relax_round": parent, "status": "completed",
            "input_path": f"E:/uploads/MLIP-round-0001_{version}/Search-group-{group:04d}/"
                          f"DFT-round-{round_index:04d}_{operation}/DFT-single-point/submission/{task_id}/task.json"}
    state.setdefault("tasks", []).append(task)
    result = {"task_id": task_id, "structure_id": "S-"+task_id, "branch_id": "B-"+task_id,
              "model_version": version, "stage": "dft_single_point", "status": "completed", "converged": True,
              "outputs": {"structure": structure.as_dict(), "composition": structure.composition.as_dict(),
                          "actual_phase": "O3", "energy": -8., "training_energy": -8., "energy_unit": "eV",
                          "forces": dft_forces.tolist(), "forces_unit": "eV/angstrom", "atom_count": atoms,
                          "training_ready": True, "training_frame_index": 0, "final_frame_index": 0,
                          "final_frame_valid": True}}
    result["outputs"]["magnetic_check"] = {
        "elements": species, "moments": [4.3 if el == "Fe" else 0 for el in species],
        "is_layered_oxide": True, "final_frame_index": 0}
    def predict(**kwargs):
        return {"model_version": version, "energy": -8.+energy_error, "energy_unit": "eV",
                "forces": (dft_forces+force_error).tolist(), "forces_unit": "eV/angstrom"}
    record_dft_products(state, result, predict if evaluator else None)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def product_root(state, index=0):
    return Path(list(state["dft_result_exports"].values())[index]["directory"])


def test_layout_reuses_model_and_upload_round_labels(tmp_path):
    state = {}
    add_result(state)
    phase = tmp_path / "m1" / "phase_diagram.csv"
    phase.parent.mkdir()
    phase.write_text("existing phase CSV")
    original = phase.stat().st_mtime_ns
    export_dft_products(state, tmp_path)
    root = product_root(state)
    assert root == tmp_path / "epoch0_m1/Search-group-0001/DFT-round-0001_abcdef"
    assert set(p.name for p in root.rglob("*") if p.is_file()) == {
        "training.json", "dft_records.json", "mlip_dft_metrics.json",
        "energy_comparison.csv", "force_comparison.csv", "magnetic_moments.csv", "metrics.csv",
        "energy_parity.png", "energy_parity.pdf", "energy_parity.svg",
        "force_parity.png", "force_parity.pdf", "force_parity.svg", "parity_metrics.json"}
    assert not (tmp_path / "dft_results").exists()
    assert phase.stat().st_mtime_ns == original
    assert len(read_csv(tmp_path / "epoch0_m1/round_metrics.csv")) == 3


def test_parity_rows_units_sign_species_and_final_frame(tmp_path):
    state = {}
    add_result(state)
    export_dft_products(state, tmp_path)
    root = product_root(state)
    energy = read_csv(dft_product_path(root, "energy_comparison.csv"))[0]
    assert float(energy["dft_energy_eV"]) == -8
    assert float(energy["mlip_energy_eV"]) == -7
    assert float(energy["energy_error_eV"]) == 1
    assert float(energy["dft_energy_eV_per_atom"]) == -4
    assert float(energy["mlip_energy_eV_per_atom"]) == -3.5
    assert energy["x_Na_per_O2"] == "2.0000000000"
    assert energy["frame_index"] == "0" and energy["phase"] == "O3"
    assert energy["structure_id"] == "S-T1" and energy["branch_id"] == "B-T1"
    forces = read_csv(dft_product_path(root, "force_comparison.csv"))
    assert len(forces) == 6
    assert [(r["atom_index"], r["element"], r["component"]) for r in forces] == [
        ("0", "Na", "x"), ("0", "Na", "y"), ("0", "Na", "z"),
        ("1", "O", "x"), ("1", "O", "y"), ("1", "O", "z")]
    assert float(forces[1]["dft_force_eV_per_A"]) == -.2
    assert float(forces[1]["mlip_force_eV_per_A"]) == 0
    assert all(float(r["force_error_eV_per_A"]) == pytest.approx(.2) for r in forces)
    assert (dft_product_path(root, "energy_comparison.csv")).read_bytes().startswith(b"\xef\xbb\xbf")


def test_json_csv_metrics_agree_with_recomputed_parity_errors(tmp_path):
    state = {}
    add_result(state)
    add_result(state, task_id="T2", atoms=3, energy_error=3., force_error=.4)
    export_dft_products(state, tmp_path)
    root = product_root(state)
    metrics = json.loads((dft_product_path(root, "mlip_dft_metrics.json")).read_text(encoding="utf-8"))
    summary = {r["metric"]: r for r in read_csv(dft_product_path(root, "metrics.csv"))}
    for name, file, field in (("energy_total", "energy_comparison.csv", "energy_error_eV"),
                              ("energy_per_atom", "energy_comparison.csv", "energy_error_eV_per_atom"),
                              ("forces", "force_comparison.csv", "force_error_eV_per_A")):
        errors = np.asarray([float(r[field]) for r in read_csv(dft_product_path(root, file))])
        assert float(summary[name]["mae"]) == pytest.approx(np.abs(errors).mean())
        assert float(summary[name]["rmse"]) == pytest.approx(np.sqrt(np.mean(errors**2)))
        assert metrics[name]["mae"] == float(summary[name]["mae"])
        assert metrics[name]["rmse"] == float(summary[name]["rmse"])
        assert int(summary[name]["sample_count"]) == len(errors)
        assert summary[name]["status"] == "completed"
        assert "mse" not in metrics[name]
    assert metrics["energy_total"]["mae"] == 2
    assert metrics["energy_total"]["rmse"] == pytest.approx(np.sqrt(5))
    assert metrics["forces"]["mae"] == pytest.approx(.32)
    assert metrics["forces"]["rmse"] == pytest.approx(np.sqrt(.112))


def test_missing_prediction_is_blank_not_zero_and_training_preserved(tmp_path):
    state = {}
    add_result(state, evaluator=False)
    export_dft_products(state, tmp_path)
    root = product_root(state)
    energy = read_csv(dft_product_path(root, "energy_comparison.csv"))[0]
    assert energy["dft_energy_eV"] == "-8.0"
    assert energy["mlip_energy_eV"] == energy["energy_error_eV"] == ""
    assert energy["comparison_status"] == "not_evaluated"
    assert "evaluator unavailable" in energy["reason"]
    assert len(read_csv(dft_product_path(root, "force_comparison.csv"))) == 6
    assert all(r["mlip_force_eV_per_A"] == "" for r in read_csv(dft_product_path(root, "force_comparison.csv")))
    assert all(r["mae"] == r["rmse"] == "" and r["sample_count"] == "0"
               for r in read_csv(dft_product_path(root, "metrics.csv")))
    assert len(json.loads((dft_product_path(root, "training.json")).read_text(encoding="utf-8"))) == 1


def test_zero_error_remains_a_valid_number(tmp_path):
    state = {}
    add_result(state, energy_error=0, force_error=0)
    export_dft_products(state, tmp_path)
    for row in read_csv(dft_product_path(product_root(state), "metrics.csv")):
        assert row["mae"] == row["rmse"] == "0.0"
        assert row["status"] == "completed"


def test_partial_round_reports_pending_and_uses_only_recovered_pairs(tmp_path):
    state = {}
    add_result(state)
    pending = deepcopy(state["tasks"][0])
    pending.update(task_id="T2", status="pending")
    state["tasks"].append(pending)
    export_dft_products(state, tmp_path)
    root = product_root(state)
    metrics = json.loads((dft_product_path(root, "mlip_dft_metrics.json")).read_text(encoding="utf-8"))
    assert metrics["matched_structures"] == metrics["recovered_tasks"] == 1
    assert metrics["expected_tasks"] == 2 and metrics["pending_task_ids"] == ["T2"]
    assert all(r["status"] == "partial" and r["pending_tasks"] == "1" for r in read_csv(dft_product_path(root, "metrics.csv")))
    assert len(read_csv(dft_product_path(root, "energy_comparison.csv"))) == 1


def test_unchanged_export_and_reordered_records_do_not_rewrite_or_infer(tmp_path, monkeypatch):
    state = {}
    add_result(state)
    add_result(state, task_id="T2")
    export_dft_products(state, tmp_path)
    paths = list(tmp_path.rglob("*.csv")) + list(tmp_path.rglob("*.json"))
    stamps = {path: path.stat().st_mtime_ns for path in paths}
    def forbidden(*args, **kwargs):
        raise AssertionError("export must not infer or classify")
    monkeypatch.setattr("scientific_layer.structures.load_result_structure.load_result_structure", forbidden)
    state["dft_dataset_records"].reverse()
    state["dft_training_records"].reverse()
    export_dft_products(state, tmp_path)
    assert {path: path.stat().st_mtime_ns for path in paths} == stamps


def test_new_round_does_not_rewrite_previous_round_and_summary_is_version_scoped(tmp_path):
    state = {}
    add_result(state)
    export_dft_products(state, tmp_path)
    old = product_root(state)
    stamps = {p: p.stat().st_mtime_ns for p in old.iterdir()}
    add_result(state, task_id="T2", operation="fedcba", round_index=2)
    add_result(state, task_id="T3", version="m2")
    export_dft_products(state, tmp_path)
    assert {p: p.stat().st_mtime_ns for p in old.iterdir()} == stamps
    assert len(state["dft_result_exports"]) == 3
    assert len(read_csv(tmp_path / "epoch0_m1/round_metrics.csv")) == 6
    assert len(read_csv(tmp_path / "epoch1_m2/round_metrics.csv")) == 3
    assert {r["model_version"] for r in read_csv(tmp_path / "epoch0_m1/round_metrics.csv")} == {"m1"}


@pytest.mark.parametrize("mutate", [
    lambda r: r["mlip_prediction"].update(model_version="wrong"),
    lambda r: r["mlip_prediction"].update(forces_unit="wrong"),
    lambda r: r["mlip_prediction"].update(forces=[[0, 0, 0]]),
    lambda r: r.pop("mlip_prediction"),
    lambda r: r.update(final_frame_valid=False),
    lambda r: r.update(final_frame_index=1),
    lambda r: r.update(training_ready=False),
])
def test_invalid_saved_prediction_cannot_enter_plot_pairs_or_metrics(tmp_path, mutate):
    state = {}
    add_result(state)
    mutate(state["dft_dataset_records"][0])
    export_dft_products(state, tmp_path)
    row = read_csv(dft_product_path(product_root(state), "energy_comparison.csv"))[0]
    assert row["comparison_status"] == "not_evaluated" and row["mlip_energy_eV"] == ""
    assert all(r["sample_count"] == "0" for r in read_csv(dft_product_path(product_root(state), "metrics.csv")))


def test_windows_paths_and_registry_fallback_preserve_exact_round(tmp_path):
    state = {}
    add_result(state)
    state["tasks"][0]["input_path"] = state["tasks"][0]["input_path"].replace("/", "\\")
    export_dft_products(state, tmp_path)
    assert product_root(state).name == "DFT-round-0001_abcdef"
    state["tasks"][0].pop("input_path")
    state["upload_layout"] = {"operations": {"m1:Search-group-0001:DFT": {"abcdef": 1}}}
    export_dft_products(state, tmp_path)
    assert product_root(state).name == "DFT-round-0001_abcdef"


def test_missing_round_evidence_is_labelled_unassigned_without_guessing(tmp_path):
    state = {}
    add_result(state)
    state["tasks"][0].pop("input_path")
    export_dft_products(state, tmp_path)
    assert product_root(state).name.startswith("DFT-round-unassigned-")


def test_conflicting_group_is_rejected_before_any_output(tmp_path):
    state = {}
    add_result(state)
    state["tasks"][0]["input_path"] = state["tasks"][0]["input_path"].replace("Search-group-0001", "Search-group-0002")
    with pytest.raises(ValueError, match="conflicting"):
        export_dft_products(state, tmp_path)
    assert not list(tmp_path.rglob("*.csv"))


def test_conflicting_scope_destination_is_rejected_before_any_output(tmp_path):
    state = {}
    add_result(state)
    add_result(state, task_id="T2", parent=1)
    with pytest.raises(ValueError, match="conflicting DFT scopes"):
        export_dft_products(state, tmp_path)
    assert not list(tmp_path.rglob("*.csv"))


def test_legacy_exports_are_not_moved_or_deleted(tmp_path):
    state = {}
    add_result(state)
    old = tmp_path / "dft_results/m1/DFT-round-old/training.json"
    old.parent.mkdir(parents=True)
    old.write_text("old output retained")
    export_dft_products(state, tmp_path)
    assert old.read_text() == "old output retained"
    assert product_root(state) != old.parent


def test_same_numbered_rounds_in_different_search_groups_stay_separate(tmp_path):
    state = {}
    add_result(state)
    add_result(state, task_id="T2", group=2)
    export_dft_products(state, tmp_path)
    assert len(state["dft_result_exports"]) == 2
    assert {Path(row["directory"]).parent.name for row in state["dft_result_exports"].values()} == {
        "Search-group-0001", "Search-group-0002"}
    assert len(read_csv(tmp_path / "epoch0_m1/round_metrics.csv")) == 6
