"""Model-isolated hull updates, frozen references, and MC/Relax sample semantics."""
from copy import deepcopy

import pytest

from analysis_layer.phase.update_local_mlip_hull_pool import update_local_mlip_hull_pool
from analysis_layer.phase.branch_relax_hull import hull_energy_per_atom, rank_relaxed_branches
from analysis_layer.phase.refresh_identified_phases import refresh_identified_phases
from analysis_layer.phase.update_phase_diagram import update_phase_diagram


def task(path, name, composition, energy, *, version="m1", stage="relax_and_feature"):
    return {"task_id": name, "structure_id": name, "branch_id": name,
            "model_version": version, "stage": stage, "status": "completed", "converged": True,
            "outputs": {"structure_path": str(path), "composition": composition,
                        "energy": energy, "energy_unit": "eV", "actual_phase": "O3",
                        "phase_identification": {"status": "identified", "phase": "O3"}}}


def test_mc_updates_only_its_model_and_preserves_frozen_reference(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("mock")
    config = {"system_config": {"system_id": "layered"}, "mlip": {"version": "m1"}}
    empty = {"Fe": 1, "O": 2}
    full = {"Na": 1, "Fe": 1, "O": 2}
    state = {"tasks": [task(path, "a", empty, -3), task(path, "b", full, -4),
                       task(path, "other", full, -40, version="m2")]}
    ledger = tmp_path / "pools.json"
    first = update_local_mlip_hull_pool(state, config=config, path=ledger)
    old_version = first["current_branch_hull_version"]
    old_pool = deepcopy(first["branch_hull_batches"][old_version])
    other_version = first["current_branch_hull_by_model"]["m2"]
    mc = task(path, "mc", full, -5, stage="deep_search")
    mc["converged"] = None  # MC normal completion has no Relax convergence flag.
    mc["outputs"]["actual_phase"] = "O3"
    first["tasks"].append(mc)
    first["tasks"].append({**task(path, "failed", full, -100, stage="deep_search"), "status": "failed"})
    updated = update_local_mlip_hull_pool(first, config=config, path=ledger)
    new_version = updated["current_branch_hull_version"]
    assert new_version != old_version
    assert updated["branch_hull_batches"][old_version] == old_pool
    assert updated["current_branch_hull_by_model"]["m2"] == other_version
    assert hull_energy_per_atom(updated["branch_hull_batches"][new_version], full) == pytest.approx(-1.25)
    assert hull_energy_per_atom(old_pool, full) == pytest.approx(-1)
    assert hull_energy_per_atom(old_pool, {"Li": 1}) is None
    before = ledger.read_text()
    repeated = update_local_mlip_hull_pool(updated, config=config, path=ledger)
    assert repeated["current_branch_hull_version"] == new_version
    assert ledger.read_text() == before
    switched = update_local_mlip_hull_pool(repeated,
        config={**config, "mlip": {"version": "m2"}}, path=ledger)
    assert switched["current_branch_hull_version"] == other_version


def test_mc_does_not_change_relax_sigma(tmp_path):
    path = tmp_path / "s.vasp"
    path.write_text("mock")
    comp = {"Na": 1, "Fe": 1, "O": 2}
    rows = [task(path, str(index), comp, energy) for index, energy in enumerate((-4, -3, -2))]
    for row in rows:
        row["branch_id"] = "B"
    mc = task(path, "mc", comp, -8, stage="deep_search")
    mc["converged"] = None
    mc["outputs"]["actual_phase"] = "O3"
    mc["branch_id"] = "B"
    config = {"system": {"system_id": "layered"}, "mlip": {"version": "m1"}}
    state = update_local_mlip_hull_pool({"tasks": [*rows, mc]}, config=config, path=tmp_path / "pool.json")
    ranked, _ = rank_relaxed_branches([{"branch_id": "B"}], state["branch_hull_batches"][state["current_branch_hull_version"]])
    assert ranked[0]["branch_relax_sample_count"] == 3
    assert ranked[0]["relaxed_energy_per_atom"] == -1
    assert ranked[0]["hull_reference_energy_per_atom"] == -2
    assert ranked[0]["branch_energy_std_per_atom"] == pytest.approx(0.2041241452)


def test_formal_diagrams_keep_versions_separate_and_support_electrode_endpoints(tmp_path):
    records = [{"record_id": f"{version}-{name}", "source_version": version,
                "energy_method": "mlip", "composition": composition, "energy": energy,
                "energy_unit": "eV"}
               for version, scale in (("m1", 1), ("m2", 10))
               for name, composition, energy in (("empty", {"Fe": 1, "O": 2}, -3 * scale),
                   ("full", {"Na": 1, "Fe": 1, "O": 2}, -4 * scale))]
    result = update_phase_diagram(records, active_model_version="m1", output_directory=tmp_path)
    assert set(result["mlip_by_version"]) == {"m1", "m2"}
    first = result["diagrams"]["mlip"]
    assert first["status"] == "completed"
    assert first["model_version"] == "m1"
    assert len(first["entries"]) == 2
    assert {entry["source_version"] for entry in first["entries"]} == {"m1"}
    assert first["entries"][0]["energy_per_O2"] == -3
    assert first["version"] != result["mlip_by_version"]["m2"]["version"]
    assert first["energy_basis_id"] != result["mlip_by_version"]["m2"]["energy_basis_id"]
    unknown = update_phase_diagram(records)
    assert unknown["diagrams"]["mlip"]["status"] == "unknown"
    switched = update_phase_diagram(records, active_model_version="m2")
    assert switched["diagrams"]["mlip"]["entries"][0]["energy_per_O2"] == -30


def test_phase_backfill_changes_version_without_rewriting_old_snapshot(tmp_path):
    records = [{"record_id": "a", "structure_id": "S", "phase": "O3",
                "model_version": "m1", "energy_method": "mlip",
                "composition": {"Na": 1, "Fe": 1, "O": 2},
                "energy": -4, "energy_unit": "eV"}]
    initial = update_phase_diagram(records, active_model_version="m1", output_directory=tmp_path)
    old = initial["diagrams"]["mlip"]
    history_path = tmp_path / "m1" / "history" / f"phase_diagram_mlip_{old['version']}.json"
    old_text = history_path.read_text(encoding="utf-8")
    repeated = update_phase_diagram(records, active_model_version="m1", output_directory=tmp_path,
                                    parent_versions={"mlip:m1": old["version"]})
    assert repeated["diagrams"]["mlip"] == old
    assert history_path.read_text(encoding="utf-8") == old_text
    state = {"tasks": [{"task_id": "T", "structure_id": "S", "model_version": "m1",
                         "stage": "relax_and_feature", "status": "completed",
                         "outputs": {"energy": -4, "actual_phase": "P3",
                                     "phase_identification": {"status": "identified", "phase": "P3"}}}],
             "phase_records": records}
    assert refresh_identified_phases(state)
    assert state["phase_records"][0]["phase"] == "P3"
    assert state["phase_records"][0]["source_phase"] == "O3"
    assert not refresh_identified_phases(state)
    updated = update_phase_diagram(state["phase_records"], active_model_version="m1",
                                   output_directory=tmp_path)
    assert updated["diagrams"]["mlip"]["version"] != old["version"]
    assert history_path.read_text(encoding="utf-8") == old_text
