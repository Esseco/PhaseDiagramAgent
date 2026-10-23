"""Configurable high-spin bounds and plots use actual recovered evidence."""

from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_dft_spin_acceptance import make_result, make_manager
from phase_agent.science.dft.spin_acceptance import apply_dft_spin_standard
from phase_agent.analysis.feedback.dft_magnetic_tables import magnetic_rows, magnetic_chat_summary
from phase_agent.analysis.feedback.dft_magnetic_plots import export_magnetic_plots
from phase_agent.analysis.feedback.dft_result_products import (
    record_dft_products,
    export_dft_products,
)


def test_custom_ranges_invalidate_cache_and_recover_qualification():
    manager = make_manager()
    original = make_result(moments=(-4.3, 3.9, 0, 0))
    before = deepcopy(original)
    first = apply_dft_spin_standard(original, manager)
    assert first["checks_passed"] is True
    manager.data["system_config"]["dft_magnetic_abs_ranges"] = {"Fe": [3.5, 4.0]}
    second = apply_dft_spin_standard(first, manager)
    assert second["checks_passed"] is False
    assert second["outputs"]["magnetic_check"]["anomalous_atoms"][0]["element"] == "Fe"
    assert (
        second["outputs"]["magnetic_check"]["evidence_sha256"]
        != first["outputs"]["magnetic_check"]["evidence_sha256"]
    )
    manager.data["system_config"]["dft_magnetic_abs_ranges"] = {"Fe": [4.3, 4.3]}
    third = apply_dft_spin_standard(second, manager)
    assert third["checks_passed"] is True
    assert original == before


@pytest.mark.parametrize(
    "bounds",
    [
        {"Fe": [5, 4]},
        {"NotAnElement": [1, 2]},
        {"Fe": [True, 4]},
        {"Fe": [0, float("inf")]},
        {"Fe": [1]},
        None,
    ],
)
def test_invalid_range_never_approves(bounds):
    manager = make_manager()
    manager.data["system_config"]["dft_magnetic_abs_ranges"] = bounds
    result = apply_dft_spin_standard(make_result(), manager)
    assert result["checks_passed"] is False
    assert result["outputs"]["magnetic_check"]["status"] == "unknown"


def test_round_export_plots_rejected_sites_and_actual_limits(tmp_path):
    manager = make_manager()
    manager.data["system_config"]["dft_magnetic_abs_ranges"] = {"Fe": [3.8, 4.1]}
    state = {}
    record_dft_products(state, make_result(), manager=manager)
    export_dft_products(state, tmp_path)
    export = next(iter(state["dft_result_exports"].values()))
    plots = export["magnetic_plots"]
    assert plots["status"] == "completed"
    assert plots["site_counts"] == {"Fe_absolute": 1, "Mn_absolute": 1}
    assert all(Path(path).is_file() for path in plots["files"].values())
    assert "Fe 3.8–4.1" in magnetic_chat_summary(state)
    assert "magnetic_distribution_Fe_absolute" in (tmp_path / "output_index.md").read_text(
        encoding="utf-8"
    )
    times = {key: Path(path).stat().st_mtime_ns for key, path in plots["files"].items()}
    export_dft_products(state, tmp_path)
    assert times == {key: Path(path).stat().st_mtime_ns for key, path in plots["files"].items()}


def test_other_transition_metals_vectors_and_missing(tmp_path):
    record = {
        "task_id": "a",
        "magnetic_moments": {"elements": ["Co", "Ni", "O"], "moments": [-1.7, None, 0]},
    }
    report = export_magnetic_plots(magnetic_rows([record]), tmp_path / "scalar")
    assert report["site_counts"] == {"Co_absolute": 1}
    assert report["excluded_sites"] == 1
    record["magnetic_moments"].update(noncollinear=True, moments=[[0, 0, 2], None, [0, 0, 0]])
    report = export_magnetic_plots(magnetic_rows([record]), tmp_path / "vector")
    assert report["site_counts"] == {"Co_vector_norm": 1}
    assert report["excluded_sites"] == 1
    assert export_magnetic_plots([], tmp_path / "empty")["status"] == "unavailable"


def test_nonfinite_sites_and_file_failure_do_not_break_recovery(tmp_path, monkeypatch):
    record = {
        "task_id": "a",
        "magnetic_moments": {"elements": ["Fe", "Mn"], "moments": [float("nan"), 3.5]},
    }
    rows = magnetic_rows([record])
    report = export_magnetic_plots(rows, tmp_path)
    assert report["site_counts"] == {"Mn_absolute": 1}
    assert report["excluded_sites"] == 1
    original = Path.read_text

    def inaccessible(path, *args, **kwargs):
        if path.name == "magnetic_distribution.json":
            raise PermissionError("locked manifest")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", inaccessible)
    report = export_magnetic_plots(rows, tmp_path)
    assert report["status"] == "unavailable"
    assert "locked manifest" in report["reason"]


@pytest.mark.parametrize("phase", ["O3", "spinel"])
def test_any_element_manual_range_preserves_all_sites_and_gates(phase):
    from pymatgen.core import Structure, Lattice
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    manager = make_manager()
    manager.data["system_config"]["dft_magnetic_abs_ranges"] = {"Co": [1.5, 2.0], "O": [0, 0.1]}
    structure = Structure(
        Lattice.cubic(4),
        ["Co", "Ni", "O", "Na"],
        [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
    )
    incoming = make_result()
    incoming["outputs"].update(
        structure=structure.as_dict(),
        composition=structure.composition.get_el_amt_dict(),
        actual_phase=phase,
        phase_identification={"status": "identified", "phase": phase},
        magnetic_moments={"elements": ["Co", "Ni", "O", "Na"], "moments": [-1.8, 0.8, 0.2, 0.01]},
    )
    original = deepcopy(incoming)
    result = apply_dft_spin_standard(incoming, manager)
    assert result["checks_passed"] is False
    assert result["outputs"]["magnetic_check"]["anomalous_atoms"][0]["element"] == "O"
    assert result["outputs"]["magnetic_moments"] == original["outputs"]["magnetic_moments"]
    record = {**result, **result["outputs"]}
    rows = magnetic_rows([record])
    assert [row["spin_assessment"] for row in rows] == [
        "reasonable",
        "not_checked",
        "out_of_range",
        "not_checked",
    ]
    assert [row["moment_mu_B"] for row in rows] == [-1.8, 0.8, 0.2, 0.01]
    assert spin_standard_passed(record) is False
    assert apply_dft_spin_standard(result)["checks_passed"] is False
    state = {}
    record_dft_products(state, incoming, manager=manager)
    assert (
        state["dft_dataset_records"][0]["magnetic_moments"]
        == original["outputs"]["magnetic_moments"]
    )
    manager.data["system_config"]["dft_magnetic_abs_ranges"]["O"] = [0, 0.2]
    passed = apply_dft_spin_standard(result, manager)
    assert passed["checks_passed"] is True
    assert spin_standard_passed({**passed, **passed["outputs"]}) is True
    unassessed = {**passed["outputs"]}
    unassessed.pop("spin_state_check")
    assert spin_standard_passed(unassessed) is False
    assert incoming == original


def test_manual_non_fe_missing_and_noncollinear_are_unknown():
    from pymatgen.core import Structure, Lattice

    manager = make_manager()
    manager.data["system_config"]["dft_magnetic_abs_ranges"] = {"Ni": [1, 2]}
    incoming = make_result()
    incoming["outputs"].update(
        structure=Structure(Lattice.cubic(4), ["Ni"], [[0, 0, 0]]).as_dict(),
        composition={"Ni": 1},
        actual_phase="metal",
        phase_identification={"status": "identified", "phase": "metal"},
        magnetic_check={},
        magnetic_moments={"elements": ["Ni"], "moments": None},
    )
    assert (
        apply_dft_spin_standard(incoming, manager)["outputs"]["spin_state_check"]["status"]
        == "unknown"
    )
    incoming["outputs"]["magnetic_moments"].update(moments=[[0, 0, 1.5]], noncollinear=True)
    result = apply_dft_spin_standard(incoming, manager)
    assert result["outputs"]["spin_state_check"]["status"] == "unknown"
    assert result["checks_passed"] is False
    record = {**result, **result["outputs"]}
    assert magnetic_rows([record])[0]["spin_assessment"] == "unknown"


def test_unconstrained_legacy_elements_keep_original_moments():
    from pymatgen.core import Structure, Lattice

    incoming = make_result()
    incoming["outputs"].update(
        structure=Structure(Lattice.cubic(4), ["Ni", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]).as_dict(),
        composition={"Ni": 1, "O": 1},
        magnetic_check={"elements": ["Ni", "O"], "moments": [-0.8, 0.01]},
    )
    result = apply_dft_spin_standard(incoming, make_manager())
    assert result["checks_passed"] is True
    rows = magnetic_rows([{**result, **result["outputs"]}])
    assert [row["moment_mu_B"] for row in rows] == [-0.8, 0.01]
    assert all(row["spin_assessment"] == "not_checked" for row in rows)
    assert all(row["expected_abs_min_mu_B"] is None for row in rows)
