import gzip
import json

import pytest
from pymatgen.core import Lattice, Structure

from Process_AL_MC.structure_check import LayeredFrameworkReference, check_structure
from tests.test_phase_framework_review import slab
from tests.test_exact_mc_occupations import fake_runner


def test_distinct_coincident_atoms_and_short_periodic_self_image():
    overlap = Structure(Lattice.cubic(5), ["Co", "O"], [[0, 0, 0], [0, 0, 0]])
    result = check_structure(overlap)
    assert result["status"] == "invalid"
    assert result["minimum_distance"] == 0
    assert "minimum_distance" in result["failed_checks"]
    tiny = Structure(Lattice.orthorhombic(0.4, 10, 10), ["Co"], [[0, 0, 0]])
    assert check_structure(tiny)["status"] == "invalid"


def test_numerical_and_composition_failures_precede_neighbors(monkeypatch):
    s = slab()
    monkeypatch.setattr(
        Structure, "get_neighbor_list", lambda *a, **k: pytest.fail("neighbor call")
    )
    assert check_structure(s, energy_per_atom=float("nan"))["status"] == "invalid"
    reference = s.composition.as_dict()
    s.replace(0, "Fe")
    assert check_structure(s, expected_composition=reference)["status"] == "invalid"


def test_cached_reference_and_tm_exchange(monkeypatch):
    s = slab()
    s.make_supercell([2, 1, 1])
    s.replace(1, "Fe")
    calls = []
    neighbors = Structure.get_neighbor_list

    def counted(structure, *args, **kwargs):
        calls.append(id(structure))
        return neighbors(structure, *args, **kwargs)

    monkeypatch.setattr(Structure, "get_neighbor_list", counted)
    original = s.copy()
    reference = LayeredFrameworkReference(original)
    initial_calls = calls.count(id(original))
    assert reference.status == "ready"
    s[0], s[1] = s[1].specie, s[0].specie
    # Reference construction is the only time initial neighbors are computed.
    assert check_structure(s, reference=reference, check_framework=True)["status"] == "passed"
    assert check_structure(s, reference=reference, check_framework=True)["status"] == "passed"
    assert calls.count(id(original)) == initial_calls
    final = Structure(Lattice.hexagonal(2.8, 1), slab().species, slab().frac_coords)
    result = check_structure(
        final, reference=LayeredFrameworkReference(slab()), check_framework=True
    )
    assert result["status"] == "invalid"
    json.dumps(result, allow_nan=False)


def test_uncertain_reference_is_explicit():
    s = Structure(Lattice.cubic(12), ["Co", "O", "O"], [[0, 0, 0], [0.2, 0, 0], [0.7, 0, 0]])
    result = check_structure(s, reference=LayeredFrameworkReference(s), check_framework=True)
    assert result["status"] == "uncertain"
    assert not result["framework"]["auto_confirmed"]


def test_invalid_trial_is_counted_but_never_updates_chain_best_or_pool(tmp_path):
    runner = fake_runner(tmp_path / "run", patience=None)
    runner.max_steps = 2
    evaluate = runner._relax_and_evaluate

    def changed(structure, traj_dir=None):
        final, data, frames = evaluate(structure, traj_dir)
        if len(runner.evaluated) == 2:
            final.replace(1, final[1].specie, final[0].frac_coords)
            data["energy_mean_per_atom"] = -100.0
        return final, data, frames

    runner._relax_and_evaluate = changed
    runner.run()
    assert runner.actual_mc_steps == 2
    assert runner.best_energy == -1.0
    assert all(item["energy"] == -1.0 for item in runner.pool)
    assert len(runner.pool) == 2
    assert runner.trace[0]["status"] == "structure_rejected"
    assert runner.trace[1]["status"] == "normal"
    assert runner.structure_check_counts["rejected"] == 1
    assert runner.last_improve_step == 0
    status = json.loads(gzip.decompress((runner.out_dir / "status.json.gz").read_bytes()))
    assert status["structure_check_counts"]["rejected"] == 1
    runner._history.close()


def test_invalid_initial_structure_never_enters_pool(tmp_path):
    runner = fake_runner(tmp_path / "run")
    evaluate = runner._relax_and_evaluate

    def bad(structure, traj_dir=None):
        final, data, frames = evaluate(structure, traj_dir)
        final.replace(1, final[1].specie, final[0].frac_coords)
        return final, data, frames

    runner._relax_and_evaluate = bad
    with pytest.raises(ValueError, match="initial structure"):
        runner.run()
    assert not runner.pool and runner.best_energy is None
    if runner._history is not None:
        runner._history.close()


def test_basic_check_avoids_distance_matrix(monkeypatch):
    monkeypatch.setattr(
        Structure, "distance_matrix", property(lambda s: pytest.fail("dense matrix"))
    )
    assert check_structure(slab())["status"] == "passed"


def test_noncompetitive_small_energy_change_skips_framework(tmp_path, monkeypatch):
    runner = fake_runner(tmp_path / "run")
    runner._prepare_structure_check(runner.structure)
    runner.current_eval = {"energy_mean_per_atom": -1.0}
    runner.pool_size = 1
    runner.pool = [{"energy": -2.0}]
    calls = []

    def inspect(s):
        calls.append(1)
        return {"status": "no_clear_destruction", "auto_confirmed": False}

    monkeypatch.setattr(runner._framework_reference, "inspect", inspect)
    assert (
        runner._check_candidate(runner.structure, {"energy_mean_per_atom": -0.99})["scope"]
        == "basic"
    )
    assert not calls
    assert runner._check_candidate(runner.structure, {"energy_mean_per_atom": -0.94})[
        "energy_jump_triggered"
    ]
    assert len(calls) == 1


def test_rejected_trials_obey_patience_and_occupation_state(tmp_path):
    runner = fake_runner(tmp_path / "run", patience=1)
    initial_occupation = []
    evaluate = runner._relax_and_evaluate

    def changed(structure, traj_dir=None):
        final, data, frames = evaluate(structure, traj_dir)
        if len(runner.evaluated) == 1:
            initial_occupation.extend(runner.na_occupation)
        else:
            final.replace(1, final[1].specie, final[0].frac_coords)
            data["energy_mean_per_atom"] = -100.0
        return final, data, frames

    runner._relax_and_evaluate = changed
    runner.run()
    assert runner.actual_mc_steps == 1 and runner.stop_reason == "patience"
    assert runner.na_occupation == initial_occupation
    assert len(runner.pool) == 1 and runner.best_energy == -1.0
    runner._history.close()


def test_disordered_structure_is_unknown_not_confirmed_invalid():
    s = Structure(Lattice.cubic(5), [{"Co": 0.5, "Fe": 0.5}], [[0, 0, 0]])
    assert check_structure(s)["status"] == "uncertain"
    assert LayeredFrameworkReference(s).status == "uncertain"


def test_remote_backend_carries_shared_checker(tmp_path):
    from phase_agent.tools.remote.manual_upload_runner import _bundle_mc_backend

    _bundle_mc_backend(tmp_path)
    assert (tmp_path / "mc_backend/Process_AL_MC/structure_check.py").is_file()


def test_simple_stage_rejects_overlap_before_branch_identification(monkeypatch):
    from phase_agent.science.structures import run_structure_check as stage

    monkeypatch.setattr(
        stage, "identify_branch_parameters", lambda *a, **kw: pytest.fail("branch check")
    )
    s = slab()
    s.replace(1, s[1].specie, s[0].frac_coords)
    result = stage.run_structure_check(s, {}, {})
    assert result["status"] == "completed" and result["converged"] is False


def test_joint_ordering_rejects_bad_candidate_and_continues(tmp_path):
    from Process_AL_MC import plan_ordering_search
    from Process_AL_MC.ordering.history import OccupationStore
    from Process_AL_MC.ordering.search import run_chain
    from pymatgen.core import Lattice, Structure

    s = Structure(
        Lattice.hexagonal(3, 10),
        ["Na", "Na", "Ni", "Ni", "O", "O", "O", "O"],
        [
            [0, 0, 0.15],
            [0.5, 0.5, 0.65],
            [0, 0, 0.4],
            [0.5, 0.5, 0.9],
            [0, 0.5, 0.3],
            [0.5, 0, 0.5],
            [0, 0.5, 0.8],
            [0.5, 0, 0],
        ],
    )
    plan = plan_ordering_search(
        s,
        phase="P2",
        tm_ratios={"Ni": 1, "Mn": 1},
        na_content="1/2",
        h_matrices=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        model={"version": "fake", "model_path": "/remote/fake.model", "sha256": "0" * 64},
        options={"max_steps": 3, "min_distance": 0, "chains_per_H": 1, "patience": 20},
    )
    matrix = plan["matrices"][0]
    calls = []

    def evaluate(structure, context):
        calls.append(1)
        if len(calls) == 2:
            structure.replace(1, structure[1].specie, structure[0].frac_coords)
        return {
            "structure": structure.as_dict(),
            "energy_per_atom": -100 if len(calls) == 2 else -1,
            "phase": "P2",
            "converged": True,
        }

    store = OccupationStore(tmp_path / "history.sqlite", {"case": "invalid"})
    summary = run_chain(
        plan, matrix, matrix["initial_occupations"][0], 0, tmp_path, store, evaluate
    )
    bad = [row["result"] for row in store.records() if row["result"]["energy_per_atom"] == -100]
    assert len(calls) >= 3 and summary["actual_new_mc_steps"] == 3
    assert bad and all(not row["ranking_eligible"] for row in bad)
    assert all(row["structure_check"]["status"] == "invalid" for row in bad)
