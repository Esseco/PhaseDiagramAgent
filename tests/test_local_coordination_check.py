import json

import numpy as np
import pytest
from pymatgen.core import Structure

from Process_AL_MC.structure_check import (
    LayeredFrameworkReference,
    check_structure,
    compact_structure_check,
)
from tests.test_phase_framework_review import slab
from tests.test_exact_mc_occupations import fake_runner


def two_site_reference():
    initial = slab()
    initial.make_supercell([2, 1, 1])
    return initial, LayeredFrameworkReference(initial)


def test_adjacent_persistent_coordination_damage_is_confirmed(monkeypatch):
    _, reference = two_site_reference()
    monkeypatch.setattr(reference, "_coordination_counts", lambda neighbors: np.array([[5, 5]] * 3))
    result = reference._inspect_coordination(None)
    assert result["auto_confirmed"] and result["affected_tm_count"] == 2
    assert result["status"] == "changed_confirmed"


@pytest.mark.parametrize(
    "counts",
    [
        [[5, 6], [5, 6], [5, 6]],  # only one site, not an adjacent pair
        [[5, 5], [6, 7], [7, 7]],  # direction changes as cutoff grows
        [[5, 5], [6, 6], [6, 6]],  # ordinary cutoff sensitivity
    ],
)
def test_isolated_or_cutoff_sensitive_changes_are_not_rejected(monkeypatch, counts):
    _, reference = two_site_reference()
    monkeypatch.setattr(reference, "_coordination_counts", lambda neighbors: np.asarray(counts))
    result = reference._inspect_coordination(None)
    assert result["status"] == "uncertain" and not result["auto_confirmed"]


def test_unstable_initial_coordination_cannot_confirm_damage(monkeypatch):
    _, reference = two_site_reference()
    reference._initial_coordination[:, 0] = 5
    monkeypatch.setattr(reference, "_coordination_counts", lambda neighbors: np.array([[5, 5]] * 3))
    result = reference._inspect_coordination(None)
    assert result["eligible_tm_count"] == 1 and not result["auto_confirmed"]


def test_one_tm_periodic_cell_is_checked(monkeypatch):
    reference = LayeredFrameworkReference(slab())
    monkeypatch.setattr(
        reference, "_coordination_counts", lambda neighbors: np.array([[7], [7], [7]])
    )
    assert reference._inspect_coordination(None)["auto_confirmed"]


def test_local_check_reuses_topology_neighbors_and_compacts_evidence(monkeypatch):
    initial, reference = two_site_reference()
    calls = []
    neighbors = Structure.get_neighbor_list

    def counted(structure, radius, *args, **kwargs):
        calls.append(radius)
        return neighbors(structure, radius, *args, **kwargs)

    monkeypatch.setattr(Structure, "get_neighbor_list", counted)
    result = check_structure(initial, reference=reference, check_framework=True)
    assert result["status"] == "passed"
    assert result["framework"]["local_coordination"]["status"] == "preserved"
    assert len(calls) == 2  # basic short cutoff + one shared topology/local table
    compact = compact_structure_check(result)
    assert "final_counts" not in compact["framework"]["local_coordination"]
    json.dumps(compact, allow_nan=False)


def test_mc_rejects_local_damage_then_continues_without_pool_pollution(tmp_path):
    runner = fake_runner(tmp_path / "run", patience=None)
    runner.max_steps = 2
    evaluate = runner._relax_and_evaluate

    def evaluated(structure, traj_dir=None):
        final, data, frames = evaluate(structure, traj_dir)
        if len(runner.evaluated) == 2:
            original = runner._framework_reference.inspect
            runner._framework_reference.inspect = lambda s: {
                "status": "coordination_damage_confirmed",
                "auto_confirmed": True,
                "local_coordination": {"status": "changed_confirmed", "auto_confirmed": True},
            }
            runner.restore_inspect = original
            data["energy_mean_per_atom"] = -100
        elif len(runner.evaluated) == 3:
            runner._framework_reference.inspect = runner.restore_inspect
        return final, data, frames

    runner._relax_and_evaluate = evaluated
    runner.run()
    assert runner.actual_mc_steps == 2 and runner.best_energy == -1
    assert runner.trace[0]["status"] == "structure_rejected"
    assert runner.trace[0]["structure_check"]["failed_checks"] == ["local_coordination"]
    assert runner.trace[1]["status"] == "normal"
    assert len(runner.pool) == 2 and all(item["energy"] == -1 for item in runner.pool)
    runner._history.close()


def test_phase_cache_fingerprint_binds_structure_check_rule(monkeypatch):
    from phase_agent.analysis.phase import phase_quality_reports as reports

    monkeypatch.setattr(reports, "diagnose_phase_diagram", lambda *a, **kw: {"status": "completed"})
    state = {
        "active_model_version": "test",
        "phase_diagrams": {"mlip": {"entries": [], "version": "v"}},
    }
    first = reports.get_phase_quality_report(state)
    monkeypatch.setattr(reports, "STRUCTURE_CHECK_RULE", "different-coordination-rule")
    second = reports.get_phase_quality_report(state)
    assert first["input_fingerprint"] != second["input_fingerprint"]
