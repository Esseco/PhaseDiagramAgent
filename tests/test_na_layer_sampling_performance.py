import hashlib
from types import SimpleNamespace
import pytest
from pymatgen.core import Lattice, Structure


def fixture_data():
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ref = Structure(
        Lattice.cubic(10),
        ["Na", "Na", "Na", "Ni", "Ni", "Ni", "O", "O", "O", "O", "O", "O"],
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.4],
            [0.1, 0.1, 0.7],
            [0.3, 0.3, 0.1],
            [0.3, 0.3, 0.4],
            [0.3, 0.3, 0.7],
            [0.5, 0.5, 0.05],
            [0.5, 0.5, 0.15],
            [0.5, 0.5, 0.35],
            [0.5, 0.5, 0.45],
            [0.5, 0.5, 0.65],
            [0.5, 0.5, 0.75],
        ],
    )
    boundary = {"P": ["O3"], "H": {"O3": [h]}, "TM_ratio": {"Ni": 1}}
    return h, ref, boundary


def test_grid_excludes_nonzero_compositions_insufficient_to_fill_layers():
    from phase_agent.science.structures.enumerate_legal_frameworks import enumerate_legal_frameworks
    from phase_agent.policies.initial_coverage import contract

    h, ref, boundary = fixture_data()
    frameworks = enumerate_legal_frameworks(boundary, {"O3": ref})
    assert frameworks[0]["allowed_x"] == ["0", "1"]
    manager = SimpleNamespace(data={"branches": {}, "structures": {}}, boundary=boundary)
    assert contract(manager, {"O3": ref}, {})["frameworks"][0]["allowed_x"] == ["0", "1"]


def test_composition_does_not_jump_multiple_na_after_filtering(monkeypatch):
    from phase_agent.science.structures import generate_composition_branches as module
    from phase_agent.science.structures import enumerate_legal_frameworks as enumerator

    h, _, _ = fixture_data()
    parent = {"P": "O3", "H": h, "x": "0", "T": ["Ni"] * 3}
    manager = SimpleNamespace(boundary={}, data={"branches": {"p": parent}})
    framework = {"P": "O3", "H": h, "allowed_x": ["0", "1"], "oxygen_count": 6}
    monkeypatch.setattr(enumerator, "enumerate_legal_frameworks", lambda *a, **k: [framework])
    monkeypatch.setattr(
        module, "generate_branch_structure", lambda *a, **k: pytest.fail("must not jump three Na")
    )
    assert (
        module.generate_composition_branches(
            manager, [framework], {}, parent_branch_ids=["p"], quota=10, seed=1
        )
        == []
    )


@pytest.mark.parametrize(
    "h",
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[2, 1, 0], [0, 2, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
        [[1, 0, 1], [0, 2, 0], [0, 0, 1]],
    ],
)
def test_metadata_layer_count_matches_expanded_geometry(h):
    from phase_agent.science.structures.rank_na_orderings_by_electrostatics import (
        count_supercell_na_layers,
        _na_layer_groups,
    )

    _, ref, _ = fixture_data()
    full = ref.copy()
    full.make_supercell(h)
    na = [i for i, s in enumerate(full) if s.is_ordered and s.specie.symbol == "Na"]
    assert count_supercell_na_layers(ref, h) == len(_na_layer_groups(full, na))


def test_inheritance_enforces_nonzero_layer_rule_and_exempts_zero(tmp_path):
    from phase_agent.science.structures.inherit_near_hull_occupancy import inherited_na_structures

    h, ref, boundary = fixture_data()
    parent = ref.copy()
    parent.remove_sites([1, 2])
    path = tmp_path / "parent.vasp"
    parent.to(filename=str(path), fmt="poscar")
    source = {
        "phase": "O3",
        "branch_H": h,
        "branch_x": "1/3",
        "structure_path": str(path),
        "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    branch = {"P": "O3", "H": h, "x": "2/3", "T": ["Ni"] * 3, "near_hull_parent_evidence": source}
    assert inherited_na_structures(branch, ref, boundary, {"O3": ref}, 3, 0) == []
    branch["x"] = "0"
    rows = inherited_na_structures(branch, ref, boundary, {"O3": ref}, 3, 0)
    assert len(rows) == 1 and rows[0][0].composition.get("Na", 0) == 0


def test_existing_invalid_input_is_blocked_and_na_zero_exempt(tmp_path):
    from phase_agent.tools.relax_mc_batch.plan import _validate_initial_na_layers

    _, ref, _ = fixture_data()
    full = tmp_path / "full.vasp"
    ref.to(filename=str(full), fmt="poscar")
    bad = ref.copy()
    bad.remove_sites([1, 2])
    source = tmp_path / "bad.vasp"
    bad.to(filename=str(source), fmt="poscar")
    with pytest.raises(ValueError, match="na_layer_occupancy_invalid"):
        _validate_initial_na_layers(source, full, "1/3", "S-bad")
    _validate_initial_na_layers(full, full, "1", "S-good")
    _validate_initial_na_layers(source, None, "0", "S-zero")


def test_dedup_only_loads_matching_branch_history(monkeypatch):
    from phase_agent.science.structures import deduplicate_candidates as module

    h, ref, _ = fixture_data()
    other = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    row = lambda H, ids: {"P": "O3", "H": H, "x": "1", "T": ["Ni"] * 3, "structure_ids": ids}
    manager = SimpleNamespace(
        data={
            "branches": {"b1": row(h, ["s1"]), "b2": row(other, ["s2"])},
            "structures": {
                "s1": {"arrangement": {"V": [1, 0]}, "id": "s1"},
                "s2": {"arrangement": {"V": [1, 0]}, "id": "s2"},
            },
        }
    )
    loaded = []

    def load(record):
        loaded.append(record["id"])
        return ref.copy()

    monkeypatch.setattr(module, "_load_historical_structure", load)

    class Matcher:
        def fit(self, a, b):
            return False

    candidate = {
        **row(h, []),
        "candidate_id": "new",
        "structure": ref,
        "arrangement": {"V": [0, 1]},
    }
    out = module.deduplicate_candidates([candidate], manager=manager, matcher=Matcher())
    assert out["statistics"]["unique"] == 1 and loaded == ["s1"]


def test_dedup_matching_occupancy_needs_no_structure_load(monkeypatch):
    from phase_agent.science.structures import deduplicate_candidates as module

    h, ref, _ = fixture_data()
    branch = {"P": "O3", "H": h, "x": "1", "T": ["Ni"] * 3, "structure_ids": ["s1"]}
    manager = SimpleNamespace(
        data={"branches": {"b": branch}, "structures": {"s1": {"arrangement": {"V": [1, 0]}}}}
    )
    monkeypatch.setattr(
        module,
        "_load_historical_structure",
        lambda *_: pytest.fail("vector match needs no structure"),
    )
    candidate = {**branch, "structure": ref, "arrangement": {"V": [1, 0]}}
    out = module.deduplicate_candidates([candidate], manager=manager)
    assert out["statistics"]["configuration_duplicates"] == 1


def test_relax_mc_preview_blocks_registered_invalid_structure(tmp_path):
    from tests.test_relax_mc_batch import fixture
    from phase_agent.tools.relax_mc_batch.plan import diagnose

    manager, state, config, context, action = fixture(tmp_path, count=1, single_x=True)
    bid = action["target_ids"][0]
    branch = manager.data["branches"][bid]
    sid = branch["structure_ids"][0]
    record = manager.data["structures"][sid]
    _, full, _ = fixture_data()
    bad = full.copy()
    bad.remove_sites([1, 2])
    bad.to(filename=record["source_path"], fmt="poscar")
    full.to(filename=record["full_na_structure_path"], fmt="poscar")
    branch["x"] = "1/3"
    record["composition"] = bad.composition.as_dict()
    with pytest.raises(ValueError, match="na_layer_occupancy_invalid"):
        diagnose([bid], state, manager, config, "v1", phase_references=context["phase_references"])


def test_fixed_tm_cache_reuses_identical_candidates_without_aliasing(monkeypatch, tmp_path):
    from phase_agent.science.structures import generate_branch_structure as g

    h, ref, boundary = fixture_data()
    path = tmp_path / "ref.vasp"
    ref.to(filename=str(path), fmt="poscar")
    calls = []

    def fake(structure, x, states):
        calls.append(str(x))
        return structure

    monkeypatch.setattr(g, "_assign_na", fake)
    kw = {
        "boundary": boundary,
        "phase": "O3",
        "H": h,
        "x": "1",
        "phase_references": {"O3": path},
        "preserve_reference_tm": True,
    }
    with g.sampling_structure_cache() as stats:
        a = g.generate_branch_structure(**kw, seed=1)
        b = g.generate_branch_structure(**kw, seed=2)
        assert a.as_dict() == b.as_dict() and a is not b
        a.remove_sites([0])
        c = g.generate_branch_structure(**kw, seed=3)
    assert len(c) == len(b) and len(calls) == 1 and stats["hits"] == 2


def test_cache_preserves_real_electrostatic_structure(tmp_path):
    from phase_agent.science.structures import generate_branch_structure as g

    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ref = Structure(
        Lattice.cubic(5),
        ["Na", "Na", "Ni", "Ni", "O", "O", "O", "O"],
        [
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.1],
            [0.2, 0.2, 0.2],
            [0.7, 0.7, 0.2],
            [0.3, 0.3, 0.3],
            [0.8, 0.8, 0.3],
            [0.3, 0.3, 0.6],
            [0.8, 0.8, 0.6],
        ],
    )
    path = tmp_path / "ref.vasp"
    ref.to(filename=str(path), fmt="poscar")
    boundary = {"P": ["O3"], "H": {"O3": [h]}, "TM_ratio": {"Ni": 1}}
    kw = {
        "boundary": boundary,
        "phase": "O3",
        "H": h,
        "x": "1/2",
        "phase_references": {"O3": path},
        "preserve_reference_tm": True,
        "oxidation_states": {"Na": 1, "Ni": 3.5, "O": -2},
    }
    before = g.generate_branch_structure(**kw, seed=42)
    with g.sampling_structure_cache() as stats:
        first = g.generate_branch_structure(**kw, seed=1)
        second = g.generate_branch_structure(**kw, seed=2)
    assert before.as_dict() == first.as_dict() == second.as_dict()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_region_state_copy_does_not_copy_large_immutable_evidence(monkeypatch):
    import phase_agent.tools.workflows.run_region_generation as module

    class Large:
        def __deepcopy__(self, memo):
            raise AssertionError("unrelated evidence deep copied")

    state = {
        "large_evidence": Large(),
        "branch_region_review": {"pending_task_key": "task"},
        "generation_history": [{"old": True}],
    }
    params = {
        "regions": [],
        "defer_reason": "test",
        "sampling_controls": module.CONTROLS,
        "region_preview": {},
    }
    monkeypatch.setattr(module, "preview", lambda *a, **k: {})
    result = module.run_region_generation(
        action={"task_key": "task", "parameters": params},
        context={
            "effective_config": {},
            "event_state": state,
            "manager": object(),
            "phase_references": {},
        },
    )
    assert (
        result["status"] == "completed"
        and result["state"]["branch_region_review"]["status"] == "completed"
    )
    assert state["branch_region_review"] == {"pending_task_key": "task"}


def test_pending_view_only_copies_pending_mapping():
    from phase_agent.runtime.local_agent_control import _scientific_pending_view

    class Large:
        def __deepcopy__(self, memo):
            raise AssertionError("irrelevant evidence deep copied")

    state = {
        "pending_execution_policies": {"__search_mode__": {"enabled": True}, "plan": {"x": 1}},
        "evidence": Large(),
    }
    view = _scientific_pending_view(state)
    assert set(view["pending_execution_policies"]) == {"plan"}
    assert "__search_mode__" in state["pending_execution_policies"]
    assert view["evidence"] is state["evidence"]


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "paused_for_review", "summary": {"registered_structures": 3}},
        {
            "status": "completed",
            "summary": {
                "registered_structures": 3,
                "sampling_report": {"status": "paused_for_review"},
            },
        },
        {
            "status": "completed",
            "summary": {"registered_structures": 3, "initialization_failed_branches": 1},
        },
    ],
)
def test_paused_or_failed_initialization_cannot_propose_next_inputs(payload):
    from phase_agent.tools.workflows.propose_followup import propose_after_generation

    frame = {"execution_mode": "interactive", "agent_client": lambda *_: None}
    result = {
        "status": "completed",
        "state": {},
        "events": [
            {
                "action": {"tool": "generate_branches"},
                "execution": {"status": "completed", "result": payload},
            }
        ],
    }

    def forbidden(*args, **kwargs):
        pytest.fail("must not propose downstream inputs")

    assert propose_after_generation(frame, result, event_loop=forbidden) is result
