import pytest
from phase_agent.analysis.phase.build_composition_hull_entries import build_composition_hull_entries
from phase_agent.analysis.phase import build_composition_hull_entries as module


def test_hull_cache_reuses_only_unchanged_geometry_and_keeps_results(tmp_path, monkeypatch):
    source = tmp_path / "one.vasp"
    source.write_bytes(b"first")
    calls = []

    def check(path, composition, *, structure_data=None):
        calls.append((path, bool(composition.get("Na"))))
        return {
            "uniform": None,
            "occupied": source.read_bytes() == b"first",
            "status": "checked",
            "rule": "test",
        }

    monkeypatch.setattr(module, "check_na_layer_uniformity", check)
    records = [
        {
            "record_id": "zero",
            "composition": {"Na": 0, "Co": 1, "O": 2},
            "normalized_total_energy": -5,
            "original_energy": -5,
            "phase": "O1",
        },
        {
            "record_id": "one",
            "composition": {"Na": 1, "Co": 1, "O": 2},
            "normalized_total_energy": -6,
            "original_energy": -6,
            "phase": "O3",
            "structure_path": str(source),
        },
    ]
    cache = {}
    first = build_composition_hull_entries(records, layer_cache=cache)
    second = build_composition_hull_entries(records, layer_cache=cache)
    assert first == second
    assert sum(bool(na) for _, na in calls) == 1
    source.write_bytes(b"changed")
    third = build_composition_hull_entries(records, layer_cache=cache)
    assert third[1]["na_layer_occupied"] is False
    assert sum(bool(na) for _, na in calls) == 2
    fresh = build_composition_hull_entries(records)
    assert third == fresh
