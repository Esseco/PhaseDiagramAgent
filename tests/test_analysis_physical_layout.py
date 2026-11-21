from pathlib import Path
import os
import pytest

from phase_agent.analysis.feedback.physical_layout import organize_physical_layout, managed_alias


def sample(tmp_path):
    root = tmp_path / "analysis_outputs"
    paths = [
        "epoch0/phase_diagram/versions/v/report.json",
        "epoch0/phase_diagram/_internal/a.json",
        "epoch0/phase_diagrams/mlip/data.json",
        "epoch0/DFT_Selection/plot.png",
        "epoch0/Search-group-0001/DFT-round-0001/data.csv",
        "epoch0/direct_sampling/receipt.json",
        "out_data/versions/epoch0/data.csv",
        "out_data/latest/data.csv",
        "latest/out_data/data.csv",
    ]
    for name in paths:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
    return root, paths


def test_physical_moves_keep_old_paths_and_repeat_safely(tmp_path):
    root, paths = sample(tmp_path)
    before = {name: (root / name).read_bytes() for name in paths}
    result = organize_physical_layout(root)
    assert result["moved_directories"] == 9
    assert result["verified_files"] == len(paths)
    assert (root / "history/epoch0/phase_diagram/versions/v/report.json").is_file()
    assert (root / "epoch0/dft/Search-group-0001/DFT-round-0001/data.csv").is_file()
    for row in result["compatibility_aliases"]:
        source, target = Path(row["source"]), Path(row["target"])
        assert managed_alias(root, source)
        if os.name == "nt":
            assert source.lstat().st_file_attributes & 2
            assert not target.stat().st_file_attributes & 2
    assert {name: (root / name).read_bytes() for name in paths} == before
    assert organize_physical_layout(root)["moved_directories"] == 0


def test_conflict_preflight_does_not_move_anything(tmp_path):
    root, paths = sample(tmp_path)
    (root / "history/epoch0/phase_diagrams").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        organize_physical_layout(root)
    assert not (root / "epoch0/DFT_Selection").is_symlink()
    assert all((root / name).is_file() for name in paths)


def test_failed_alias_rolls_back_every_completed_move(tmp_path, monkeypatch):
    import phase_agent.analysis.feedback.physical_layout as module

    root, paths = sample(tmp_path)
    original = module._alias
    calls = []

    def fail_second(target, source):
        calls.append(source)
        if len(calls) == 2:
            raise OSError("injected alias failure")
        original(target, source)

    monkeypatch.setattr(module, "_alias", fail_second)
    with pytest.raises(OSError, match="injected"):
        organize_physical_layout(root)
    assert all((root / name).read_bytes() == name.encode() for name in paths)
    assert not (root / "epoch0/DFT_Selection").is_junction()


def test_new_publication_after_physical_move(tmp_path):
    from phase_agent.analysis.feedback.versioned_products import publish_bundle, validate_bundle

    root = tmp_path / "analysis_outputs"
    root.mkdir()

    def build(folder):
        (folder / "data.json").write_text("{}")

    args = dict(
        kind="phase_diagram", epoch="epoch0", model="m", required=["data.json"], build=build
    )
    first = publish_bundle(root, identity={"hull": "a"}, **args)
    organize_physical_layout(root)
    second = publish_bundle(root, identity={"hull": "b"}, **args)
    assert second["status"] == "completed"
    assert validate_bundle(Path(first["directory"]))["identity"]["hull"] == "a"
    assert validate_bundle(root / "latest/phase_diagram")["identity"]["hull"] == "b"


def test_failed_epoch_inventory_preserves_alias_contents(tmp_path):
    from phase_agent.analysis.feedback.failed_epoch_archive import _inventory

    root, _ = sample(tmp_path)
    before = _inventory(root / "epoch0")
    organize_physical_layout(root)
    after = _inventory(root / "epoch0")
    # New canonical DFT/diagnostics entries also appear, but every old immutable path remains readable.
    indexed = {row["path"]: row for row in after}
    assert all(indexed[row["path"]] == row for row in before)


def test_old_project_history_is_brought_back_into_analysis_history(tmp_path):
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

    root = tmp_path / "analysis_outputs"
    root.mkdir()
    old = analysis_metadata_directory(root) / "Previous_Latest"
    old.mkdir(parents=True)
    (old / "report.json").write_bytes(b"saved report")
    organize_physical_layout(root)
    assert (root / "history/project/Previous_Latest/report.json").read_bytes() == b"saved report"
    assert (old / "report.json").read_bytes() == b"saved report"
    assert organize_physical_layout(root)["moved_directories"] == 0


def test_new_round_does_not_split_unmigrated_group(tmp_path):
    from phase_agent.analysis.feedback.output_paths import dft_round_directory

    old = tmp_path / "Search-group-0001"
    (old / "round1").mkdir(parents=True)
    assert dft_round_directory(tmp_path, old.name, "round2") == old / "round2"


def test_unregistered_alias_is_rejected(tmp_path):
    from phase_agent.analysis.feedback.physical_layout import _alias

    root = tmp_path / "analysis_outputs"
    source = root / "epoch0" / "phase_diagrams"
    source.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    _alias(outside, source)
    try:
        with pytest.raises(ValueError, match="unrecognized"):
            organize_physical_layout(root)
    finally:
        if source.is_junction():
            source.rmdir()
        else:
            source.unlink()


def test_merge_dft_summary_with_existing_history_alias(tmp_path):
    from phase_agent.analysis.feedback.physical_layout import _alias
    from phase_agent.analysis.feedback.versioned_products import publish_bundle, validate_bundle

    root = tmp_path / "analysis_outputs"
    old = root / "epoch0/dft_error"
    history = root / "history/epoch0/dft_error/versions"
    history.mkdir(parents=True)
    old.mkdir(parents=True)
    _alias(history, old / "versions")
    kwargs = dict(kind="dft_error", epoch="epoch0", model="m", required=["data.json"],
                  build=lambda folder: (folder / "data.json").write_bytes(b"{}"))
    first = publish_bundle(root, identity={"input": "old"}, **kwargs)
    original = Path(first["manifest_path"]).read_bytes()
    result = organize_physical_layout(root)
    assert result["moved_directories"] == 1
    assert managed_alias(root, old)
    summary = root / "epoch0/dft/summary"
    assert managed_alias(root, summary / "versions")
    assert Path(first["manifest_path"]).read_bytes() == original
    assert validate_bundle(summary / "latest")["version"] == first["version"]
    second = publish_bundle(root, identity={"input": "new"}, **kwargs)
    assert Path(second["latest_directory"]) == summary / "latest"
    assert validate_bundle(old / "latest")["version"] == second["version"]
    assert organize_physical_layout(root)["moved_directories"] == 0
