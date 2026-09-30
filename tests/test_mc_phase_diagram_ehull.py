"""MC allocations use the same per-atom Ehull as the current phase CSV."""

import pytest

from analysis_layer.phase.branch_relax_hull import build_relax_hull, rank_relaxed_branches
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from decision_layer.strategy.estimate_branch_mc_budget import estimate_branch_mc_budget


def _evidence():
    row = {"branch_id": "B1", "structure_id": "S1", "structure_path": "final.vasp",
           "composition": {"Na": 1, "Fe": 1, "O": 2}, "energy": -4.0,
           "energy_unit": "eV", "converged": True, "model_version": "m1"}
    pool = build_relax_hull([row], model_version="m1")
    diagram = {"method": "mlip", "status": "completed", "model_version": "m1",
               "version": "phase-v1", "entries": [{
                   "structure_id": "S1", "structure_path": "final.vasp",
                   "normalized_total_energy": -4.0, "phase_identification_status": "identified",
                   "phase": "O3", "ehull": 0.018, "ehull_unit": "eV/atom"}]}
    candidate = {"branch_id": "B1", "P": "O3", "x": "1/2", "atom_count": 4}
    return pool, diagram, candidate


def test_preview_reads_phase_diagram_value_not_local_pool_gap():
    pool, diagram, candidate = _evidence()
    old, _ = rank_relaxed_branches([candidate], pool)
    assert old[0]["relaxed_ehull"] == pytest.approx(0)
    ranked, missing = rank_relaxed_branches([candidate], pool, phase_diagram=diagram)
    assert missing == []
    assert ranked[0]["relaxed_ehull"] == pytest.approx(0.018)
    assert ranked[0]["ehull_source"] == "phase_diagram"
    config = default_layered_search_config()
    preview = estimate_branch_mc_budget([candidate], pool,
        {"phase_diagrams": {"mlip": diagram}}, config, step_limit=1000, seed=7)
    assert preview["phase_diagram_version"] == "phase-v1"
    assert preview["allocations"][0]["relaxed_ehull"] == pytest.approx(0.018)
    assert preview["allocations"][0]["phase_diagram_version"] == "phase-v1"
    assert preview["allocations"][0]["ehull_source"] == "phase_diagram"
    assert preview["allocations"][0]["relaxed_ehull_unit"] == "eV/atom"


def test_missing_or_wrong_version_phase_ehull_cannot_be_used():
    pool, diagram, candidate = _evidence()
    diagram["entries"][0]["normalized_total_energy"] = -3.0
    ranked, missing = rank_relaxed_branches([candidate], pool, phase_diagram=diagram)
    assert ranked == [] and missing == ["B1"]
    diagram["model_version"] = "m2"
    with pytest.raises(ValueError, match="模型版本"):
        rank_relaxed_branches([candidate], pool, phase_diagram=diagram)


def test_preview_requires_current_diagram_and_freezes_its_version():
    pool, diagram, candidate = _evidence()
    config = default_layered_search_config()
    with pytest.raises(ValueError, match="相图未就绪"):
        estimate_branch_mc_budget([candidate], pool, {}, config, step_limit=1000, seed=7)
    first = estimate_branch_mc_budget([candidate], pool,
        {"phase_diagrams": {"mlip": diagram}}, config, step_limit=1000, seed=7)
    diagram["version"] = "phase-v2"
    second = estimate_branch_mc_budget([candidate], pool,
        {"phase_diagrams": {"mlip": diagram}}, config, step_limit=1000, seed=7)
    assert first["allocations"][0]["task_key"] != second["allocations"][0]["task_key"]
    assert first["allocation_checksum"] != second["allocation_checksum"]
