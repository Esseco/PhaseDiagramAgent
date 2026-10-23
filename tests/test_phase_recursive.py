from copy import deepcopy
from itertools import combinations
import random

import pytest

from phase_agent.analysis.phase.phase_quality import (
    diagnose_phase_diagram,
    _points,
    _hull,
    _influence,
)
from phase_agent.analysis.phase.phase_hull_sensitivity import HullSensitivity, joint_cases
from phase_agent.analysis.phase.phase_quality_reports import review_markdown
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
from phase_agent.analysis.phase.export_current_phase_diagram import export_current_phase_diagram
from phase_agent.science.structures.boundary_utils import allowed_phases_at_x
from tests.test_phase_quality import sample, add


CO_RULE = {
    "at_x": {"0": ["O1"], "1": ["O3", "P3", "P2", "OP2"]},
    "intermediate": ["O3", "P3", "P2", "OP2"],
}


def test_simultaneous_high_groups_and_newly_exposed_layers_are_conditional():
    state = sample()
    add(state, "second-low", -0.3, x=0.25)
    add(state, "hidden", -0.2)
    before = deepcopy(state)
    report = diagnose_phase_diagram(state)
    assert state == before
    first = report["analysis_layers"][0]
    assert len(first["joint_group_ids"]) >= 2
    assert first["joint_step_lift_mev_per_O2"] == pytest.approx(200)
    hidden = next(r for r in report["structures"] if "hidden" in r["record_ids"])
    # A far backup is only considered after the current support has been removed.
    assert hidden["analysis_layer"] > 0
    assert hidden["geometry_key"] not in first["joint_geometry_keys"]
    assert hidden["priority_mev_per_O2"] == 0
    assert any(e["D_mev_per_O2"] > 100 for e in hidden["conditional_evidence"])
    assert hidden["tier"] == 2 and not hidden["automatic_exclusion_eligible"]
    assert first["scope"] == "counterfactual_only_not_rejection"


@pytest.mark.parametrize("count", [2, 3])
def test_masked_pairs_and_triples_are_found_without_large_single_effect(count):
    state = sample((0, -0.2, 0.05, 0))
    add(state, "mask2", -0.199, phase="P3")
    if count == 3:
        add(state, "mask3", -0.198, phase="P2")
    report = diagnose_phase_diagram(state)
    cases = report["analysis_layers"][0]["joint_cases"]["cases"]
    target = next(c for c in cases if c["mode"] == "geometry" and c["order"] == count)
    assert target["hull_lift_mev_per_O2"] == pytest.approx(200)
    assert target["minimal_high_impact"]
    assert set(target["record_ids"]) == {"r1", "mask2"} | ({"mask3"} if count == 3 else set())


@pytest.mark.parametrize("seed", range(12))
def test_pruned_search_matches_exhaustive_minimal_subsets(seed):
    rng = random.Random(seed)
    points = [
        {
            "key": str(i),
            "x": x,
            "energy_mev_per_O2": rng.uniform(-100, 100),
            "rows": [{"record_id": str(i)}],
        }
        for i, x in enumerate([0, 0, 0.2, 0.4, 0.4, 0.6, 0.8, 1, 1])
    ]
    hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in points])
    expected = set()
    for size in (1, 2, 3):
        for combo in combinations([p["key"] for p in points], size):
            keys = frozenset(combo)
            if any(prior < keys for prior in expected):
                continue
            lift = _influence(points, hull, keys)
            if lift is not None and lift >= 10:
                expected.add(keys)
    actual = joint_cases(points, {p["key"]: [p] for p in points}, threshold=10, keep_per_order=1000)
    found = {frozenset(c["unit_ids"]) for c in actual["cases"] if c["mode"] == "geometry"}
    assert found == expected
    evaluator = HullSensitivity(points)
    for size in (1, 2, 3):
        for combo in combinations([p["key"] for p in points], size):
            assert evaluator.lift(combo) == pytest.approx(_influence(points, hull, set(combo)))


def test_large_smooth_hull_prunes_cubic_combinations():
    count = 160
    points = [
        {
            "key": str(i),
            "x": i / (count - 1),
            "energy_mev_per_O2": 1000 * (i / (count - 1) - 0.5) ** 2,
            "rows": [{"record_id": str(i)}],
        }
        for i in range(count)
    ]
    result = joint_cases(points, {p["key"]: [p] for p in points}, threshold=10)
    assert not result["cases"]
    assert result["stats"]["geometry"]["queries"] < count * 20


def test_depth_limit_is_explicit_and_endpoint_domain_never_faked():
    report = diagnose_phase_diagram(sample(), policy={"recursive_max_layers": 1})
    assert report["recursive_truncated"] and report["recursive_stop_reason"] == "depth_limit"
    for layer in report["analysis_layers"]:
        assert "G-" not in "".join(layer["joint_geometry_keys"])
    with pytest.raises(ValueError):
        diagnose_phase_diagram(sample(), policy={"recursive_max_layers": 1.5})


def test_co_rule_is_bidirectional_and_legacy_other_system_unaffected():
    assert allowed_phases_at_x(CO_RULE, 0) == {"O1"}
    for x in ["1/1000", 0.5, 1]:
        assert "O1" not in allowed_phases_at_x(CO_RULE, x)
    assert allowed_phases_at_x(["O3", "O1"], 0) == {"O3", "O1"}


def test_outside_scope_records_retained_as_rejections_and_stale_export_blocked(tmp_path):
    state = sample()
    state["confirmed_config"] = {"system": {"boundary": {"P": CO_RULE}}}
    state["phase_diagrams"]["mlip"]["entries"][0]["phase"] = "O1"
    add(state, "bad-O1", -0.9, phase="O1")
    add(state, "bad-Na0", -0.8, phase="O3", x=0)
    records = [
        {
            **r,
            "energy": r["eform_per_O2"],
            "energy_unit": "eV",
            "energy_method": "mlip",
            "model_version": "m1",
            "status": "completed",
        }
        for r in state["phase_diagrams"]["mlip"]["entries"]
    ]
    before = deepcopy(records)
    output = update_phase_diagram(
        records, active_model_version="m1", output_state=state, include_dft=False
    )
    assert records == before
    diagram = output["diagrams"]["mlip"]
    assert {r["record_id"] for r in diagram["entries"]} == {"r0", "r1", "r2", "r3"}
    assert {r["record_id"] for r in diagram["rejected"]} == {"bad-O1", "bad-Na0"}
    assert diagram["phase_boundary_rule"] == CO_RULE
    with pytest.raises(ValueError, match="边界"):
        export_current_phase_diagram(state, directory=tmp_path)


def test_markdown_contains_combo_paths_conditional_layer_and_records():
    state = sample()
    for i, row in enumerate(state["phase_diagrams"]["mlip"]["entries"]):
        row["structure_path"] = f"/inspect/structure-{i}.vasp"
    add(state, "mask", -0.399, phase="P3")
    report = diagnose_phase_diagram(state)
    text = review_markdown(report, bundle_id="b")
    assert "最多5项" in text and "联合抬升" in text
    assert "/inspect/structure-1.vasp" in text and "r1" in text
    assert "结构地址" in text and "后层依赖前层假设" in text
