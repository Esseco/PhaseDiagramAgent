from pathlib import Path
import pytest
from analysis_layer.feedback.dft_comparison_tables import build_comparison_tables
from analysis_layer.feedback.dft_parity_plots import export_parity_plots
from tests.test_dft_comparison_csv import add_result


def tables():
    state = {}
    add_result(state)
    return build_comparison_tables(state["dft_dataset_records"], state["dft_mlip_comparisons"], round_name="round")


def test_parity_metrics_components_cache_and_change(tmp_path):
    energy, force = tables()
    result = export_parity_plots(energy, force, tmp_path, "epoch0 (m1)")
    assert result["status"] == "completed"
    assert result["metrics"]["energy"]["mae"] == 500
    assert result["metrics"]["energy"]["rmse"] == 500
    assert result["metrics"]["energy"]["rmse_unit"] == "meV/atom"
    assert result["metrics"]["force"]["points"] == 6
    assert result["metrics"]["force"]["rmse"] == pytest.approx(200)
    assert result["metrics"]["force"]["rmse_unit"] == "meV/Å"
    path = Path(result["files"]["energy_png"])
    modified = path.stat().st_mtime_ns
    assert export_parity_plots(energy, force, tmp_path, "epoch0 (m1)") == result
    assert path.stat().st_mtime_ns == modified
    energy[0]["mlip_energy_eV_per_atom"] += .1
    assert export_parity_plots(energy, force, tmp_path, "epoch0 (m1)")["fingerprint"] != result["fingerprint"]


def test_no_pairs_are_not_presented_as_zero_error(tmp_path):
    result = export_parity_plots([], [], tmp_path, "epoch0")
    assert result["status"] == "unavailable"
    assert not list(tmp_path.glob("*.png"))


def test_near_zero_force_is_warning_not_data_exclusion(tmp_path):
    energy, force = tables()
    for row in force:
        row["mlip_force_eV_per_A"] = 0
    result = export_parity_plots(energy, force, tmp_path, "epoch0")
    assert result["metrics"]["force"]["points"] == 6
    assert result["diagnostics"]["force_rms_ratio"] == 0
    assert result["diagnostics"]["warning"]


def test_path_change_reuses_plots(tmp_path):
    import shutil
    energy, force = tables()
    energy[0]["structure_path"] = "old/input.vasp"
    old = tmp_path / "old"
    result = export_parity_plots(energy, force, old, "epoch0")
    new = tmp_path / "new"
    shutil.move(str(old), str(new))
    energy[0]["structure_path"] = "new/input.vasp"
    path = new / "energy_parity.png"
    before = path.stat().st_mtime_ns
    reused = export_parity_plots(energy, force, new, "epoch0")
    assert reused["fingerprint"] == result["fingerprint"]
    assert Path(reused["files"]["energy_png"]) == path
    assert path.stat().st_mtime_ns == before


def test_publication_failure_does_not_crash_or_overwrite(tmp_path, monkeypatch):
    from analysis_layer.feedback import export_dft_products as publisher
    from matplotlib.figure import Figure
    original = Figure.savefig

    def memory_only(self, destination, **kwargs):
        assert hasattr(destination, "write")
        return original(self, destination, **kwargs)

    monkeypatch.setattr(Figure, "savefig", memory_only)
    path = tmp_path / "energy_parity.png"
    path.write_bytes(b"original")

    def denied(*args):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(publisher, "_publish_bytes", denied)
    energy, force = tables()
    report = export_parity_plots(energy, force, tmp_path, "epoch0")
    assert report["status"] == "unavailable"
    assert "发布失败" in report["reason"]
    assert path.read_bytes() == b"original"
