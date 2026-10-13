import json
import gzip

import pytest
from pymatgen.core import Lattice, Structure
from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass
from Process_AL_MC.occupation_history import pack_occupation, site_layout, decode_snapshot
from phase_agent.science.mc.project_occupation_history import history_seed
from phase_agent.science.mc.occupation_database import OccupationHistory
from phase_agent.tools.remote.run_mlip_batch import _worker_environment
from phase_agent.tools.remote.manual_upload_runner import _bundle_mc_backend
from phase_agent.tools.remote.transfer_bundle import _result_files


def full_template():
    return Structure(
        Lattice.cubic(12),
        ["Na"] * 4 + ["Co", "O", "O"],
        [
            [0, 0, 0],
            [0.25, 0, 0],
            [0.5, 0, 0],
            [0.75, 0, 0],
            [0, 0.5, 0.5],
            [0.2, 0.5, 0.5],
            [0.7, 0.5, 0.5],
        ],
    )


class FakeMC(LayeredOxide_MCOrderingClass):
    def _load_models(self, **kwargs):
        return [object()]

    def _build_relaxer(self):
        return None

    def _relax_and_evaluate(self, structure, traj_dir=None):
        self.evaluated.append(structure.copy())
        if getattr(self, "fail_trial", False) and len(self.evaluated) > 1:
            raise RuntimeError("fake model failure")
        result = {
            "n_atoms": len(structure),
            "energy_var_per_atom": 0.0,
            "energy_per_model_total": [-float(len(structure))],
            "energy_mean_per_atom": -1.0,
            "energy_std_per_atom": 0.0,
            "energy_per_model_per_atom": [-1.0],
            "force_uncertainty_per_atom": 0.0,
            "force_uncertainty_var_per_atom": 0.0,
            "force_uncertainty_max": 0.0,
            "force_uncertainty_var_max": 0.0,
        }
        return structure.copy(), result, []


def fake_runner(directory, *, seed=None, patience=20, limit=100):
    full = full_template()
    structure = full.copy()
    structure.remove_sites([3, 2, 1])
    runner = FakeMC(
        structure=structure,
        full_na_structure=full,
        mode="Na_MC_input",
        out_dir=directory,
        occupation_history_scope="B",
        occupation_history_seed=seed,
        max_steps=200,
        max_duplicate_trials=limit,
        patience=patience,
        save_every=100,
    )
    runner.evaluated = []
    return runner


def test_compact_bitmap_and_cross_model_round_persistence(tmp_path):
    key = pack_occupation(["Na"] * 50 + ["Vac"] * 50)
    assert len(key) == 13
    h = OccupationHistory(tmp_path / "seen.sqlite", "B", 100, "basis")
    assert h.add(key) and not h.add(key)
    seed = h.seed()
    h.close()
    second = OccupationHistory(tmp_path / "new_model.sqlite", "B", 100, "basis", seed=seed)
    assert key in second and len(second) == 1
    assert decode_snapshot(second.snapshot(delta=True))[1] == []
    other = OccupationHistory(tmp_path / "new_model.sqlite", "other", 100, "basis")
    assert key not in other
    second.close()
    other.close()


def test_site_numbering_change_is_rejected(tmp_path):
    h = OccupationHistory(tmp_path / "history.sqlite", "B", 4, "original")
    h.close()
    with pytest.raises(ValueError, match="ordering changed"):
        OccupationHistory(tmp_path / "history.sqlite", "B", 4, "changed")


def test_seeded_duplicates_stop_before_any_trial_model_call(tmp_path):
    n, basis = site_layout(full_template())
    history = OccupationHistory(tmp_path / "project.sqlite", "B", n, basis)
    for index in range(n):
        history.add(pack_occupation(["Na" if i == index else "Vac" for i in range(n)]))
    seed = history.seed()
    history.close()
    runner = fake_runner(tmp_path / "run", seed=seed, limit=7)
    runner.run()
    # Initialization is allowed to evaluate the seed with the current model.
    assert len(runner.evaluated) == 1
    assert runner.actual_mc_steps == 0 and runner.last_improve_step == 0
    assert runner.duplicate_rejections == 7
    status = json.loads(gzip.decompress((runner.out_dir / "status.json.gz").read_bytes()))
    assert status["stop_reason"] == "duplicate_trial_limit"
    assert (runner.out_dir / "mc_occupations.bin").is_file()
    runner._history.close()


def test_duplicates_do_not_advance_patience_or_effective_steps(tmp_path):
    runner = fake_runner(tmp_path / "run", patience=2)
    runner.run()
    assert runner.actual_mc_steps == 2
    assert len(runner.evaluated) == 3
    assert runner.stop_reason == "patience"
    _, records = decode_snapshot((runner.out_dir / "mc_occupations.bin").read_bytes())
    assert len(records) == 3
    runner._history.close()


def test_failed_evaluation_does_not_poison_history(tmp_path):
    runner = fake_runner(tmp_path / "run")
    runner.fail_trial = True
    with pytest.raises(RuntimeError, match="fake model failure"):
        runner.run()
    assert len(runner.visited) == 1
    assert runner.actual_mc_steps == 0
    runner._history.close()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_duplicate_limit_must_be_positive_integer(tmp_path, limit):
    runner = fake_runner(tmp_path / "run", limit=limit)
    with pytest.raises(ValueError, match="positive integer"):
        runner.run()


def test_project_import_is_compact_idempotent_and_shared_across_models(tmp_path):
    template = full_template()
    path = tmp_path / "full.vasp"
    template.to(filename=path, fmt="poscar")
    # Use the serialized reference, exactly as the remote backend sees it.
    count, basis = site_layout(Structure.from_file(path))
    remote = OccupationHistory(tmp_path / "remote.sqlite", "B", count, basis)
    key = pack_occupation(["Vac", "Na", "Vac", "Vac"])
    remote.add(key)
    result = tmp_path / "results"
    result.mkdir()
    remote.write_delta(result / "mc_occupations.bin")
    remote.close()
    task = {
        "task_id": "first",
        "stage": "deep_search",
        "branch_id": "B",
        "status": "completed",
        "model_version": "old",
        "result_path": str(result / "result.json"),
    }
    config = {"state_path": str(tmp_path / "workflow_state/state.json")}
    seed = history_seed(config, "B", path, tasks=[task])
    assert history_seed(config, "B", path, tasks=[task]) == seed
    fresh = OccupationHistory(tmp_path / "current_model.sqlite", "B", count, basis, seed=seed)
    assert key in fresh and len(fresh) == 1
    fresh.close()


def test_backend_and_delta_are_in_portable_handoff(tmp_path):
    _bundle_mc_backend(tmp_path)
    package = tmp_path / "mc_backend/Process_AL_MC"
    assert (package / "MC_sample.py").is_file() and (package / "occupation_history.py").is_file()
    env = _worker_environment(tmp_path)
    assert env["PYTHONPATH"].startswith(str(tmp_path / "mc_backend"))
    (tmp_path / "mc_occupations.bin").write_bytes(b"compact")
    (tmp_path / "mc_seen.sqlite").write_bytes(b"not transferred")
    assert tmp_path / "mc_occupations.bin" in _result_files(tmp_path)
    assert tmp_path / "mc_seen.sqlite" not in _result_files(tmp_path)


def test_optional_reference_file_and_no_reference_mode(tmp_path):
    import base64

    count, basis = site_layout(full_template())
    history = OccupationHistory(tmp_path / "project.sqlite", "B", count, basis)
    for index in range(count):
        history.add(pack_occupation(["Na" if i == index else "Vac" for i in range(count)]))
    reference = tmp_path / "reference.bin"
    reference.write_bytes(base64.b64decode(history.seed()))
    history.close()
    runner = fake_runner(tmp_path / "referenced", limit=5)
    runner.occupation_history_scope = None
    runner.occupation_reference_file = reference
    runner.run()
    assert runner.actual_mc_steps == 0
    assert runner.stop_reason == "duplicate_trial_limit"
    assert not (runner.out_dir / "mc_seen.sqlite").exists()
    plain = fake_runner(tmp_path / "plain", patience=1)
    plain.occupation_history_scope = None
    plain.run()
    assert plain.actual_mc_steps == 1
    assert plain._history is None
