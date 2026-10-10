import gzip
import json

import pytest

from phase_agent.persistence.ledger.branch_energy_pool_ledger import load_branch_energy_pools, save_branch_energy_pool
from phase_agent.science.mlip.mace_worker import _actual_mc_steps, _final_member_energies, _qbc_from_summary
from phase_agent.science.qbc.summarize_member_energies import summarize_member_energies


def write_gzip(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


def test_gz_pool_uses_final_frame_three_model_energies(tmp_path):
    pool = tmp_path / "pool"; pool.mkdir()
    row = {"relax_traj_file": "rank_000_relax_traj.json"}
    write_gzip(pool / "rank_000_relax_traj.json.gz", [
        {"step": 0, "energy_per_model_per_atom": [-1, -2, -3]},
        {"step": "final", "energy_per_model_per_atom": [-3.0, -3.1, -2.9]},
    ])
    assert _final_member_energies(tmp_path, row) == [-3.0, -3.1, -2.9]
    qbc = summarize_member_energies(_final_member_energies(tmp_path, row))
    assert qbc["status"] == "completed"
    assert qbc["energy_per_atom_std"] == pytest.approx(0.081649658)
    assert summarize_member_energies([-1, -2])["status"] == "insufficient_committee"


def test_actual_steps_come_from_compressed_trace(tmp_path):
    write_gzip(tmp_path / "trace.json.gz", [{"step": 1}, {"step": 14}, {"step": "final"}])
    assert _actual_mc_steps(tmp_path) == 14


def test_energy_pool_json_ledger_is_version_isolated_and_reloadable(tmp_path):
    path = tmp_path / "pools.json"
    base = {"system_id": "layered", "model_version": "m1", "energy_basis": "total_eV",
            "version": "batch-1", "records": [{"energy": -3.0}]}
    save_branch_energy_pool(base, path)
    save_branch_energy_pool({**base, "model_version": "m2", "version": "batch-2"}, path)
    loaded = load_branch_energy_pools(path, system_id="layered", mlip_version="m1",
                                      energy_basis="total_eV")
    assert loaded == [base]


def test_single_model_has_no_uncertainty_and_four_members_use_summary():
    assert _qbc_from_summary({"energy_std_per_atom": 0.0}, 1)["status"] == "not_available_single_model"
    qbc = _qbc_from_summary({"energy_std_per_atom": .02, "energy_var_per_atom": .0004,
                             "force_uncertainty_per_atom": .1,
                             "force_uncertainty_max": .3}, 4)
    assert qbc["status"] == "completed"
    assert qbc["member_count"] == 4
    assert qbc["f_std_max"] == .3
