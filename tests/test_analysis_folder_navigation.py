from phase_agent.analysis.feedback.folder_navigation import publish_folder_navigation
from phase_agent.analysis.feedback.output_paths import dft_round_directory
import pytest


def test_navigation_preserves_all_products_and_user_notes(tmp_path):
    root = tmp_path / "analysis_outputs"
    epoch = root / "epoch0"
    names = [
        "phase_diagram",
        "dft_error",
        "phase_diagrams",
        "DFT_Selection",
        "Search-group-0001",
        "outlier data",
        "dft",
    ]
    for name in names:
        path = epoch / name / "data.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"scientific evidence")
    (root / "README.md").write_text("user note", encoding="utf8")
    (epoch / "README.md").write_text("existing report", encoding="utf8")
    before = {p: p.read_bytes() for p in epoch.rglob("*.csv")}
    publish_folder_navigation(root)
    first = (epoch / "README.md").read_text(encoding="utf8")
    assert "existing report" in first
    assert all(name in first for name in names)
    assert "user note" in (root / "README.md").read_text(encoding="utf8")
    publish_folder_navigation(root)
    assert (epoch / "README.md").read_text(encoding="utf8") == first
    assert {p: p.read_bytes() for p in epoch.rglob("*.csv")} == before


def test_dft_directory_new_legacy_and_conflict(tmp_path):
    legacy = tmp_path / "Search-group-0001" / "DFT-round-0001"
    current = tmp_path / "dft" / "Search-group-0001" / "DFT-round-0001"
    assert dft_round_directory(tmp_path, legacy.parent.name, legacy.name) == current
    legacy.mkdir(parents=True)
    assert dft_round_directory(tmp_path, legacy.parent.name, legacy.name) == legacy
    current.mkdir(parents=True)
    with pytest.raises(ValueError, match="duplicate"):
        dft_round_directory(tmp_path, legacy.parent.name, legacy.name)


def test_registered_missing_dft_round_keeps_its_path(tmp_path):
    from phase_agent.analysis.feedback.export_dft_products import _product_directory

    state = {"upload_layout": {"model_rounds": {"m": 1}}}
    scope = {"model_version": "m", "search_group_index": 1}
    old = tmp_path / "epoch0" / "Search-group-0001" / "DFT-round-unassigned-id"
    state["dft_result_exports"] = {"id": {"directory": str(old)}}
    assert _product_directory(tmp_path, state, scope, [], "id") == old
    assert not old.exists()
