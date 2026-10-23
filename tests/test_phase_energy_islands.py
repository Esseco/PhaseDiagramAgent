from copy import deepcopy

import pytest

from phase_agent.analysis.phase.phase_quality import (
    diagnose_phase_diagram,
    _points,
    _energy_groups,
    screening_policy,
)
from phase_agent.analysis.phase.phase_hull_sensitivity import joint_cases
from phase_agent.analysis.phase.phase_quality_reports import review_markdown
from tests.test_phase_quality import sample, add


def masked_endpoint():
    state = sample((0, -0.2, 0.02, 0))
    add(state, "zero-twin", 0, x=0)
    add(state, "tier2a", 0.03755, x=0)
    add(state, "tier2b", 0.03755, x=0)
    for i in range(6):
        add(state, f"ref{i}", 0.196 + i * 0.001, x=0)
    return state


def test_second_threshold_keeps_searching_after_twenty_and_reports_four_member_set():
    state = masked_endpoint()
    points = _points(state["phase_diagrams"]["mlip"], [])
    result = joint_cases(
        points,
        _energy_groups(points, screening_policy()),
        threshold=20,
        escalation_threshold=100,
        order=5,
    )
    target = next(
        c
        for c in result["cases"]
        if set(c["record_ids"]) == {"r0", "zero-twin", "tier2a", "tier2b"}
    )
    assert target["hull_lift_mev_per_O2"] == pytest.approx(196)
    assert 100 in target["minimal_for_thresholds_mev_per_O2"]


def test_joint_above_100_uncertain_is_human_only_and_state_unchanged():
    state = masked_endpoint()
    before = deepcopy(state)
    report = diagnose_phase_diagram(state)
    cluster = next(
        c
        for c in report["energy_islands"]
        if set(c["record_ids"]) == {"r0", "zero-twin", "tier2a", "tier2b"}
    )
    assert cluster["hull_lift_mev_per_O2"] == pytest.approx(196)
    assert cluster["reference_status"] == "available"
    assert cluster["disposition"] == "pending_human"
    assert cluster["automatic_exclusion_eligible"] is False
    targets = [r for r in report["structures"] if r["geometry_key"] in cluster["geometry_keys"]]
    assert len(targets) == 4 and all(r["review_pending"] for r in targets)
    assert all(not r["automatic_exclusion_eligible"] for r in targets)
    assert all(r["joint_hull_lift_mev_per_O2"] >= 196 - 1e-6 for r in targets)
    assert state == before


def test_off_hull_island_is_flagged_but_not_auto_removed_and_references_not_cross_phase():
    state = sample()
    add(state, "island", -0.1, x=0.25)
    for i in range(6):
        add(state, f"ref{i}", 0.25 + i * 0.001, x=0.25)
    add(state, "different-phase", -0.099, phase="P3", x=0.25)
    report = diagnose_phase_diagram(state)
    cluster = next(c for c in report["energy_islands"] if c["record_ids"] == ["island"])
    assert cluster["hull_lift_mev_per_O2"] == 0
    assert cluster["gap_to_lowest_reference_mev_per_O2"] == pytest.approx(350)
    assert cluster["reference_count"] == 6
    row = next(r for r in report["structures"] if r["record_ids"] == ["island"])
    assert row["review_pending"] and row["hull_lift_mev_per_O2"] == 0
    assert not row["automatic_exclusion_eligible"]


def test_sparse_reference_does_not_establish_error_and_nearby_normal_points_not_islands():
    state = sample((0, -0.019, 0.02, 0))
    assert not diagnose_phase_diagram(state)["energy_islands"]
    state = sample()
    report = diagnose_phase_diagram(state)
    assert report["energy_islands"][0]["reference_status"] == "insufficient_references"
    assert report["energy_islands"][0]["energy_error_confirmed"] is False


def test_exact_100_is_not_escalated_and_report_exposes_joint_identity_and_paths():
    state = masked_endpoint()
    for r in state["phase_diagrams"]["mlip"]["entries"]:
        r["structure_path"] = "/inspect/" + r["record_id"] + ".vasp"
    report = diagnose_phase_diagram(state)
    text = review_markdown(report, bundle_id="read-only")
    assert "低能离群簇" in text and "联合抬升超过100" in text
    assert "/inspect/tier2a.vasp" in text and "人工确认" in text
    assert "固定原始参考" in text
    state = sample((0, -0.1, 0.02, 0))
    report = diagnose_phase_diagram(state)
    target = next(r for r in report["structures"] if r["record_ids"] == ["r1"])
    assert not target["automatic_exclusion_eligible"]
    assert not any(
        100 in c["minimal_for_thresholds_mev_per_O2"]
        for c in report["analysis_layers"][0]["joint_cases"]["cases"]
        if c["record_ids"] == ["r1"]
    )


def test_less_than_twenty_can_be_energy_suspect_without_automatic_removal():
    report = diagnose_phase_diagram(sample((0, -0.019, 0.05, 0)))
    target = next(r for r in report["structures"] if r["record_ids"] == ["r1"])
    assert target["hull_lift_mev_per_O2"] == pytest.approx(19)
    assert target["review_recommended"] and target["energy_island_ids"]
    assert not target["automatic_exclusion_eligible"]


@pytest.mark.parametrize("cutoff,strict", [(20, False), (100, True)])
def test_dual_threshold_search_matches_exhaustive_sets_up_to_five(cutoff, strict):
    from itertools import combinations
    from phase_agent.analysis.phase.phase_quality import _hull, _influence

    state = masked_endpoint()
    points = _points(state["phase_diagrams"]["mlip"], [])
    hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in points])
    expected = set()
    for size in range(1, 6):
        for combo in combinations([p["key"] for p in points], size):
            keys = frozenset(combo)
            if any(prior < keys for prior in expected):
                continue
            lift = _influence(points, hull, keys)
            if lift is not None and (lift > cutoff if strict else lift + 1e-7 >= cutoff):
                expected.add(keys)
    result = joint_cases(
        points,
        _energy_groups(points, screening_policy()),
        threshold=20,
        escalation_threshold=100,
        order=5,
    )
    found = {
        frozenset(c["unit_ids"])
        for c in result["cases"]
        if c["mode"] == "geometry" and cutoff in c["minimal_for_thresholds_mev_per_O2"]
    }
    assert found == expected


def test_counterfactual_plot_rebases_endpoints_but_H_keeps_original_reference():
    from phase_agent.analysis.phase.phase_quality_preview import build_quality_preview

    state = masked_endpoint()
    before = deepcopy(state)
    report = diagnose_phase_diagram(state)
    cluster = next(c for c in report["energy_islands"] if len(c["record_ids"]) == 4)
    snapshot = state["phase_diagrams"]["mlip"]
    preview = build_quality_preview(snapshot, report, cluster_ids=[cluster["cluster_id"]])
    assert preview["before_count"] - preview["after_count"] == 4
    assert min(
        e for x, e in zip(preview["plot_data"]["x"], preview["plot_data"]["eform"]) if x == 0
    ) == pytest.approx(0)
    assert preview["endpoint_reference"]["energy"] == pytest.approx([196, 0])
    assert preview["influence_reference"] == "fixed_original"
    assert preview["hull_lift_mev_per_O2"] == pytest.approx(196)
    assert state == before
    snapshot["entries"][0]["eform_per_O2"] = 0.0001
    with pytest.raises(ValueError, match="改变"):
        build_quality_preview(snapshot, report, cluster_ids=[cluster["cluster_id"]])
