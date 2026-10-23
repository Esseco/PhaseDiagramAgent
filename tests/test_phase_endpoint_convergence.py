from copy import deepcopy

import pytest

from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram, _points, _energy_groups
from phase_agent.analysis.phase.phase_quality_endpoints import select_preview_keys
from phase_agent.analysis.phase.phase_quality_preview import build_quality_preview
from phase_agent.analysis.phase.phase_quality_review import exclude_records
from phase_agent.analysis.phase.phase_quality_actions import (
    apply_quality_review,
    automatic_round_review,
)
from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report
from phase_agent.analysis.phase.phase_hull_sensitivity import joint_cases
from phase_agent.analysis.phase.phase_energy_islands import energy_islands
from tests.test_phase_quality import sample, add, row_for
from tests.test_phase_quality_review import fixture


def test_sparse_endpoint_phase_overrides_direct_H_and_other_phase_substitution():
    state = sample()
    add(state, "different-phase-end", 0.7, phase="P2", x=1)
    add(state, "tied-O3", 0.001, x=1)
    report = diagnose_phase_diagram(state)
    endpoint = row_for(report, "r3")
    assert endpoint["H_mev_per_O2"] == pytest.approx(1)
    assert endpoint["endpoint_protected"] and endpoint["tier"] == 0
    assert not endpoint["automatic_exclusion_eligible"] and not endpoint["preview_excluded"]
    singleton = sample()
    add(singleton, "different-phase-end", 0.7, phase="P2", x=1)
    row = row_for(diagnose_phase_diagram(singleton), "r3")
    assert row["H_mev_per_O2"] == pytest.approx(700)
    assert row["endpoint_protected"] and not row["automatic_exclusion_eligible"]
    with pytest.raises(ValueError, match="端点"):
        build_quality_preview(
            state["phase_diagrams"]["mlip"], report, review_ids=[endpoint["review_id"]]
        )


def test_endpoint_reference_requires_separate_narrow_same_phase_band():
    state = sample()
    for i in range(6):
        add(state, f"near-{i}", 0.001 * i, x=1)
    assert row_for(diagnose_phase_diagram(state), "r3")["endpoint_protected"]
    for i in range(5):
        add(state, f"reference-{i}", 0.200 + 0.001 * i, x=1)
    report = diagnose_phase_diagram(state)
    row = row_for(report, "r3")
    assert not row["endpoint_protected"]
    key = row["geometry_key"]
    refs = [
        r for r in report["structures"] if any(v.startswith("reference-") for v in r["record_ids"])
    ]
    with pytest.raises(ValueError, match="端点"):
        build_quality_preview(
            state["phase_diagrams"]["mlip"],
            report,
            review_ids=[row["review_id"]] + [r["review_id"] for r in refs],
        )
    assert key in report["preview_excluded_geometry_keys"]
    assert not any(r["preview_excluded"] for r in refs)


def test_joint_case_cannot_be_partially_removed_when_endpoint_member_is_guarded():
    state = sample()
    add(state, "end-P2", 0.7, phase="P2", x=1)
    points = _points(state["phase_diagrams"]["mlip"], [])
    report = diagnose_phase_diagram(state)
    end, interior = row_for(report, "r3"), row_for(report)
    keys, blocked = select_preview_keys(
        points,
        report["screening_policy"],
        [
            {
                "geometry_keys": [end["geometry_key"], interior["geometry_key"]],
                "hull_lift_mev_per_O2": 700,
            }
        ],
        [],
    )
    assert not keys and end["geometry_key"] in blocked


def test_common_mutation_boundary_rejects_endpoint_before_any_side_effect():
    state, manager, *_ = fixture()
    before, ledger = deepcopy(state), deepcopy(manager.data)
    with pytest.raises(ValueError, match="端点"):
        exclude_records(
            state, manager, {"r0"}, set(), entry={"model_version": "m1", "reason": "manual"}
        )
    assert state == before and manager.data == ledger


def layered_state(count=12):
    state = sample((0, -0.08 * count, 0, 0))
    for i in range(1, count):
        add(state, f"layer-{i}", -0.08 * (count - i))
    return state


def test_default_recursion_exceeds_three_passes_and_final_right_panel_is_rechecked():
    state = layered_state()
    before = deepcopy(state)
    report = diagnose_phase_diagram(state, policy={"outlier_gap_mev_per_O2": 10000})
    assert state == before
    assert len(report["analysis_layers"]) > 4
    assert report["recursive_converged"] and not report["recursive_truncated"]
    applied = [l for l in report["analysis_layers"] if l["applied_to_preview"]]
    excluded = set(report["preview_excluded_geometry_keys"])
    assert excluded == {k for l in applied for k in l["joint_geometry_keys"]}
    assert not report["analysis_layers"][-1]["joint_geometry_keys"]
    active = [p for p in _points(state["phase_diagrams"]["mlip"], []) if p["key"] not in excluded]
    policy = report["screening_policy"]
    groups = _energy_groups(active, policy)
    cases = joint_cases(
        active, groups, threshold=50, order=5, escalation_threshold=100, strict_review=True
    )
    assert not select_preview_keys(
        active, policy, cases["cases"], energy_islands(active, groups, policy)
    )[0]
    hidden = row_for(report, "layer-10")
    assert hidden["H_mev_per_O2"] == 0 and hidden["conditional_only"]
    assert (
        hidden["preview_excluded"]
        and hidden["tier"] == 2
        and not hidden["automatic_exclusion_eligible"]
    )


def test_explicit_debug_limit_does_not_union_unapplied_last_layer_into_preview():
    state = layered_state()
    report = diagnose_phase_diagram(
        state, policy={"recursive_max_layers": 2, "outlier_gap_mev_per_O2": 10000}
    )
    assert report["recursive_truncated"] and not report["recursive_converged"]
    assert report["recursive_stop_reason"] == "depth_limit"
    assert report["analysis_layers"][0]["applied_to_preview"]
    assert not report["analysis_layers"][1]["applied_to_preview"]
    assert set(report["preview_excluded_geometry_keys"]) == set(
        report["analysis_layers"][0]["joint_geometry_keys"]
    )


def test_human_retain_remains_in_hypothetical_chain(tmp_path):
    state, manager, *_ = fixture()
    state["phase_diagrams"]["mlip"]["entries"][1]["eform_per_O2"] = -0.075
    state["phase_records"][1]["eform_per_O2"] = -0.075
    report = get_phase_quality_report(state)
    result = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row_for(report)["review_id"]],
        decision="retain",
        reason="keep",
        human_requested=True,
    )
    retained = row_for(get_phase_quality_report(result["state"]))
    assert retained["disposition"] == "human_retained"
    assert not retained["preview_excluded"]


def test_automatic_loop_continues_after_four_passes_using_only_current_evidence(
    monkeypatch, tmp_path
):
    import phase_agent.analysis.phase.phase_quality_actions as actions

    state, manager, *_ = fixture()
    batch = state["relax_mc_batches"]["b1"]
    batch.update(
        status="completed",
        model_version="m1",
        import_checksum="ok",
        result_summary={"status": "completed", "model_version": "m1"},
    )
    for task in state["tasks"]:
        task["status"] = "completed"
    state["phase_diagrams"]["mlip"]["entries"] *= 3
    state["pass"] = 0

    def report(current, **kwargs):
        return {
            "structures": []
            if current["pass"] == 6
            else [{"review_id": str(current["pass"]), "automatic_exclusion_eligible": True}]
        }

    def apply(current, shadow, report, **kwargs):
        updated = deepcopy(current)
        updated["pass"] += 1
        return {"state": updated, "excluded_record_ids": [str(current["pass"])]}

    monkeypatch.setattr(actions, "get_phase_quality_report", report)
    monkeypatch.setattr(actions, "apply_quality_review", apply)
    result = automatic_round_review(state, manager, bundle_id="b1", backup_root=tmp_path)
    assert result["state"]["pass"] == 6 and len(result["excluded_record_ids"]) == 6
