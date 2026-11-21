import random
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher

from phase_agent.science.structures.deduplicate_candidates import deduplicate_candidates
from tests.test_na_layer_sampling_performance import fixture_data


def make_candidates():
    _, ref, boundary = fixture_data()
    h = [[4, 0, 0], [0, 1, 0], [0, 0, 1]]
    boundary["H"]["O3"] = [h]
    full = ref.copy()
    full.make_supercell(h)
    sodium = [i for i, s in enumerate(full) if s.specie.symbol == "Na"]
    randomizer = random.Random(11)
    rows = []
    for i in range(16):
        selected = sorted(randomizer.sample(range(len(sodium)), 6))
        structure = full.copy()
        structure.remove_sites([sodium[j] for j in range(len(sodium)) if j not in selected])
        rows.append(
            {
                "P": "O3",
                "H": h,
                "x": "1/2",
                "T": ["Ni"] * 12,
                "structure": structure,
                "candidate_id": str(i),
                "arrangement": {"V": [int(j in selected) for j in range(len(sodium))]},
            }
        )
    translated = deepcopy(rows[0])
    translated["candidate_id"] = "translated"
    translated["structure"].translate_sites(range(len(translated["structure"])), [1, 0, 0])
    translated["arrangement"] = None
    rows.append(translated)
    near = deepcopy(rows[1])
    near["candidate_id"] = "near"
    near["structure"].translate_sites([0], [1e-8, 0, 0])
    near["arrangement"] = None
    rows.append(near)
    return boundary, ref, rows


def test_batch_matcher_preserves_full_dedup_results(tmp_path):
    boundary, ref, rows = make_candidates()
    path = tmp_path / "ref.vasp"
    ref.to(filename=str(path), fmt="poscar")
    manager = SimpleNamespace(boundary=boundary, data={"branches": {}, "structures": {}})
    baseline = deduplicate_candidates(
        deepcopy(rows),
        manager=manager,
        phase_references={"O3": path},
        matcher=StructureMatcher(primitive_cell=False, scale=False),
    )
    optimized = deduplicate_candidates(
        deepcopy(rows), manager=manager, phase_references={"O3": path}
    )
    assert optimized["duplicate_records"] == baseline["duplicate_records"]
    assert optimized["statistics"] == baseline["statistics"]
    assert [r["candidate_id"] for r in optimized["unique_candidates"]] == [
        r["candidate_id"] for r in baseline["unique_candidates"]
    ]


def test_reduction_cache_separates_inputs_and_returns_independent_copies():
    from phase_agent.science.structures.deduplicate_candidates import _BatchStructureMatcher

    _, ref, _ = fixture_data()
    matcher = _BatchStructureMatcher(primitive_cell=False, scale=False)
    a = matcher._get_reduced_structure(ref, False, True)
    original = a.as_dict()
    a.remove_sites([0])
    assert matcher._get_reduced_structure(ref, False, True).as_dict() == original
    changed = ref.copy()
    changed.translate_sites([0], [1e-8, 0, 0])
    matcher._get_reduced_structure(changed, False, True)
    assert len(matcher._reduced_cache) == 2
    matcher._get_reduced_structure(ref, False, False)
    assert len(matcher._reduced_cache) == 3


def test_cache_fallback_does_not_hide_reduction_failure(monkeypatch):
    from phase_agent.science.structures.deduplicate_candidates import _BatchStructureMatcher

    _, ref, _ = fixture_data()
    matcher = _BatchStructureMatcher(primitive_cell=False, scale=False)

    def fail(*args, **kwargs):
        raise ValueError("reduction failure")

    monkeypatch.setattr(matcher, "_uncached_reduction", fail)
    with pytest.raises(ValueError, match="reduction failure"):
        matcher._get_reduced_structure(ref, False, True)


def test_supplied_matcher_remains_in_control():
    _, ref, _ = fixture_data()

    class Matcher:
        def __init__(self):
            self.calls = 0

        def fit(self, a, b):
            self.calls += 1
            return False

    matcher = Matcher()
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    rows = [
        {
            "P": "O3",
            "H": h,
            "x": "1",
            "T": ["Ni"] * 3,
            "structure": ref,
            "candidate_id": str(i),
            "arrangement": {"V": [i, 1]},
        }
        for i in range(2)
    ]
    result = deduplicate_candidates(rows, matcher=matcher)
    assert matcher.calls == 1 and result["statistics"]["unique"] == 2
