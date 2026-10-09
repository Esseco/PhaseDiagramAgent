from copy import deepcopy
from pathlib import Path
import pytest

from analysis_layer.phase.combined_phase_diagram import build_combined_phase_diagram, refresh_combined_phase_diagram
from tests.test_dft_comparison_csv import add_result


def fixture():
    state = {}
    for task, x, error in (("left", 0, 1.), ("right", 1, 2.)):
        add_result(state, task_id=task, energy_error=error)
        row = state["dft_dataset_records"][-1]
        row.update(composition={"Na": x, "Fe": 1, "O": 2}, actual_phase="O3",
                   phase_identification={"status": "identified"}, structure_sha256=task)
    def entry(name, x, phase="O3"):
        return {"record_id": name, "structure_id": name, "structure_sha256": name,
                "composition": {"Na": x, "Fe": 1, "O": 2}, "phase": phase,
                "phase_identification_status": "identified", "original_energy": -7.,
                "normalized_total_energy": -7.}
    snapshot = {"status": "completed", "model_version": "m1",
                "entries": [entry("left", 0), entry("mid", .5), entry("right", 1)]}
    state["phase_diagrams"] = {"mlip": snapshot}
    return state, entry


def test_corrected_interpolation_and_dft_override():
    state, _ = fixture()
    result = build_combined_phase_diagram(state["phase_diagrams"]["mlip"],
        state["dft_dataset_records"], state["dft_mlip_comparisons"])
    rows = {r["record_id"]: r for r in result["entries"]}
    assert result["status"] == "completed"
    assert rows["mid"]["corrected_energy_eV"] == pytest.approx(-8.5)
    assert rows["mid"]["energy_source"] == "corrected_mlip"
    assert rows["left"]["energy_source"] == "dft"
    assert "eform_per_O2" in rows["mid"] and "ehull" in rows["mid"]


def test_no_extrapolation_or_cross_phase_correction():
    state, entry = fixture()
    state["phase_diagrams"]["mlip"]["entries"] += [entry("outside", 1.5), entry("other", .5, "P3")]
    result = build_combined_phase_diagram(state["phase_diagrams"]["mlip"],
        state["dft_dataset_records"], state["dft_mlip_comparisons"])
    assert result["status"] == "partial" and result["excluded_structures"] == 2
    for row in result["entries"]:
        if row["record_id"] in {"outside", "other"}:
            assert row["corrected_energy_eV"] is None and "ehull" not in row


def test_model_isolation_and_tm_mismatch():
    state, entry = fixture()
    records = deepcopy(state["dft_dataset_records"])
    for row in records:
        row["model_version"] = "other"
    result = build_combined_phase_diagram(state["phase_diagrams"]["mlip"], records, [])
    assert result["status"] == "unknown"
    bad = entry("bad", .5)
    bad["composition"]["Fe"] = 2
    state["phase_diagrams"]["mlip"]["entries"].append(bad)
    with pytest.raises(ValueError, match="TM/O2"):
        build_combined_phase_diagram(state["phase_diagrams"]["mlip"],
                                    state["dft_dataset_records"], state["dft_mlip_comparisons"])


def test_export_reuses_unchanged_snapshot(tmp_path):
    state, _ = fixture()
    refresh_combined_phase_diagram(state, tmp_path)
    path = Path(state["phase_diagrams"]["combined"]["csv_path"])
    assert path == tmp_path / "epoch0_m1/phase_diagrams/combined/phase_diagram.csv"
    timestamp = path.stat().st_mtime_ns
    refresh_combined_phase_diagram(state, tmp_path)
    assert path.stat().st_mtime_ns == timestamp
    assert "energy_source" in path.read_text(encoding="utf-8-sig")


def test_saved_combined_export_does_not_recalculate(tmp_path):
    from analysis_layer.phase.export_current_phase_diagram import export_current_phase_diagram
    state, entry = fixture()
    state["active_model_version"] = "m1"
    state["phase_diagrams"]["mlip"]["entries"].append(entry("outside", 2))
    refresh_combined_phase_diagram(state, tmp_path)
    path, count, version = export_current_phase_diagram(state, method="combined")
    assert path.is_file() and count == 4 and version
    state["active_model_version"] = "m2"
    with pytest.raises(ValueError, match="当前模型"):
        export_current_phase_diagram(state, method="combined")
