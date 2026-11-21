import json
import pytest
from ase import Atoms
from ase.io import write
from phase_agent.science.mlip import mace_worker


@pytest.mark.parametrize(
    "case", ["four", "four_index1", "single", "missing", "bad_mean", "bad_index"]
)
def test_mc_returns_main_energy_for_selected_committee_rank(tmp_path, monkeypatch, case):
    import Process_AL_MC

    atoms = Atoms(
        "NaCoO2", positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5], pbc=True
    )
    source = tmp_path / "input.vasp"
    write(source, atoms, format="vasp")
    values = [-5.0, -4.0, -3.0, -2.0] if case != "single" else [-5.0]
    mean = sum(values) / len(values)

    class Runner:
        MC_ENERGY_BASIS = "main_model"

        def __init__(self, **kwargs):
            self.pool = [
                {
                    "step": 2,
                    "eval": {"energy_per_model_per_atom": ([] if case == "missing" else values)},
                }
            ]
            self.relaxation_observations = []

        def run(self, **kwargs):
            folder = tmp_path / "pool"
            folder.mkdir()
            write(folder / "rank_000_step_0002.vasp", atoms, format="vasp")
            (folder / "pool_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "rank": 0,
                            "step": 2,
                            "structure_file": "rank_000_step_0002.vasp",
                            "energy_mean_per_atom": mean + (0.1 if case == "bad_mean" else 0),
                            "energy_std_per_atom": 1.0,
                            "energy_var_per_atom": 1.0,
                            "force_uncertainty_per_atom": 0.2,
                            "force_uncertainty_max": 0.7,
                        }
                    ]
                ),
                encoding="utf8",
            )
            (tmp_path / "status.json").write_text(
                json.dumps({"stop_reason": "patience"}), encoding="utf8"
            )
            return atoms

    monkeypatch.setattr(Process_AL_MC, "LayeredOxide_MCOrderingClass", Runner)
    monkeypatch.setattr(mace_worker, "_threshold_monitor_factory", lambda _: object())
    job = {
        "operation": "mc",
        "structure_path": str(source),
        "output_directory": str(tmp_path),
        "model_version": "m",
        "parameters": {
            "main_model_index": (9 if case == "bad_index" else 1 if case == "four_index1" else 0),
            "feature_difference_threshold": 3.0 if case == "four" else 1.0,
        },
        "model_paths": ["/m1"] if case == "single" else ["/m1", "/m2", "/m3", "/m4"],
    }
    if case in {"missing", "bad_mean", "bad_index"}:
        with pytest.raises(ValueError, match="mc_selected_frame_member_energy_unavailable"):
            mace_worker.run_mace_worker(job)
    else:
        result = mace_worker.run_mace_worker(job)
        expected = -4.0 if case == "four_index1" else -5.0
        assert result["energy_per_atom"] == expected
        assert result["energy"] == expected * 4
        assert result["committee_energy_per_atom"] == mean
        assert result["energy_source"] == "main_model_selected_frame"
        assert result["feature_difference_threshold"] == (3.0 if case == "four" else 1.0)
        assert result["qbc"]["member_count"] == len(values)
        if case in {"four", "four_index1"}:
            assert result["pool_summary"][0]["energy_per_model_per_atom"] == values
            assert result["qbc"]["force_rms_disagreement"] == 0.2
            assert result["qbc"]["f_std_max"] == 0.7


def test_mc_rejects_backend_that_cannot_prove_main_model_ranking(tmp_path, monkeypatch):
    import Process_AL_MC

    class LegacyRunner:
        def __init__(self, **kwargs):
            raise AssertionError("must reject before model initialization")

    monkeypatch.setattr(Process_AL_MC, "LayeredOxide_MCOrderingClass", LegacyRunner)
    monkeypatch.setattr(mace_worker, "_threshold_monitor_factory", lambda _: object())
    with pytest.raises(RuntimeError, match="mc_backend_main_model_energy_required"):
        mace_worker.run_mace_worker(
            {"operation": "mc", "output_directory": str(tmp_path), "structure_path": "unused"}
        )


def test_all_mc_batches_ship_audited_worker(tmp_path):
    from phase_agent.tools.remote.manual_upload_runner import _needs_scoped_mc_worker

    assert _needs_scoped_mc_worker(tmp_path, [{"stage": "deep_search"}])
    assert not _needs_scoped_mc_worker(tmp_path, [{"stage": "relax_and_feature"}])
