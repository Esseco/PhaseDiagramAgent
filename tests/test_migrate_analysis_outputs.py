import json
from pathlib import Path
from execution_layer.local.migrate_analysis_outputs import migrate_analysis_outputs


def test_layout_migration_preserves_data_and_updates_references(tmp_path):
    old = tmp_path / "current" / "phase_diagrams"
    root = old / "m1" / "dft_rounds" / "Search-group-0001" / "DFT-round-0001_op"
    root.mkdir(parents=True)
    source = root / "energy_comparison.csv"
    source.write_bytes(b"DFT,MLIP\n1,2\n")
    state = tmp_path / "current" / "state.json"
    state.write_text(json.dumps({"path": str(source), "directory": str(root), "energy": 1.25}))
    report = migrate_analysis_outputs(tmp_path)
    moved = tmp_path / "outputs/m1/Search-group-0001/DFT-round-0001_op/comparisons/energy_comparison.csv"
    assert moved.read_bytes() == b"DFT,MLIP\n1,2\n"
    restored = json.loads(state.read_text())
    assert Path(restored["path"]) == moved
    assert restored["energy"] == 1.25
    assert (Path(report["backup"]) / "state.json").is_file()
    assert not old.exists()
