from types import SimpleNamespace

import pytest


def test_branch_workflow_records_stage_timings_without_changing_result(monkeypatch, tmp_path):
    from phase_agent.tools.workflows import run_branch_generation as module

    manager = SimpleNamespace(data={"branches": {}, "structures": {}}, boundary={"P": ["O3"]})
    monkeypatch.setattr(module, "propose_branches", lambda *a, **kw: [])
    monkeypatch.setattr(
        module,
        "deduplicate_candidates",
        lambda rows, **kw: {
            "unique_candidates": rows,
            "statistics": {"input": 0, "unique": 0, "configuration_duplicates": 0},
        },
    )
    monkeypatch.setattr(module, "estimate_candidate_cost", lambda rows, **kw: rows)
    monkeypatch.setattr(
        module,
        "select_candidates",
        lambda rows, **kw: {"selected_candidates": rows, "rejected_candidates": [], "summary": {}},
    )
    monkeypatch.setattr(module, "register_candidate_batch", lambda *a, **kw: [])
    shared = {"sampling": 3.0}
    result = module.run_branch_generation(
        manager,
        {},
        structure_directory=tmp_path,
        quotas={},
        batch_size=1,
        initial_states_per_branch=1,
        seed=1,
        timings=shared,
    )
    assert result["registered"] == []
    values = result["summary"]["timings_seconds"]
    assert values["sampling"] == 3.0
    for key in (
        "branch_deduplication",
        "candidate_cost_and_selection",
        "initialization",
        "initial_electrostatic_ranking",
        "structure_deduplication",
        "registration",
    ):
        assert key in values and values[key] >= 0
    assert "子集" in result["summary"]["timings_note"]


def test_ranking_time_recorded_on_failure(monkeypatch):
    import phase_agent.science.structures.initialize_branch_structures as module
    from pymatgen.core import Lattice, Structure

    ref = Structure(
        Lattice.cubic(5),
        ["Na", "Ni", "O", "O"],
        [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]],
    )
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    boundary = {"P": ["O3"], "H": {"O3": [h]}, "TM_ratio": {"Ni": 1}}
    monkeypatch.setattr(
        module,
        "rank_na_orderings_by_electrostatics",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("rank failed")),
    )
    times = {}
    with pytest.raises(RuntimeError, match="静电能"):
        module.initialize_branch_structures(
            [{"P": "O3", "H": h, "x": "1", "T": ["Ni"]}],
            boundary,
            {"O3": ref},
            initial_states_per_branch=1,
            seed=1,
            timings=times,
        )
    assert times["initial_electrostatic_ranking"] >= 0
