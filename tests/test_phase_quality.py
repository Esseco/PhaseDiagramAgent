from copy import deepcopy
import json
from pathlib import Path

import pytest

from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_reports import (
    candidate_quality_evidence,
    get_phase_quality_report,
    quality_summary,
)


def sample(energies=(0, -0.4, 0.02, 0)):
    rows = [
        {
            "record_id": f"r{i}",
            "structure_id": f"s{i}",
            "structure_sha256": f"g{i}",
            "phase": "O3",
            "composition": {"Co": 1, "O": 2, "Na": x},
            "eform_per_O2": e,
            "source_version": "m1",
            "phase_identification_status": "identified",
        }
        for i, (x, e) in enumerate(zip((0, 0.5, 0.5, 1), energies))
    ]
    return {
        "active_model_version": "m1",
        "phase_diagrams": {
            "mlip": {"status": "completed", "model_version": "m1", "version": "h1", "entries": rows}
        },
    }


def row_for(report, record="r1"):
    return next(r for r in report["structures"] if record in r["record_ids"])


def add(state, name, energy, *, phase="O3", x=0.5):
    row = deepcopy(state["phase_diagrams"]["mlip"]["entries"][1])
    row.update(
        record_id=name, structure_id=name, structure_sha256=name, eform_per_O2=energy, phase=phase
    )
    row["composition"]["Na"] = x
    state["phase_diagrams"]["mlip"]["entries"].append(row)


def test_read_only_priority_is_actual_maximum_lift_without_weighting():
    state = sample()
    before = deepcopy(state)
    report = diagnose_phase_diagram(state)
    row = row_for(report)
    assert row["priority_mev_per_O2"] == pytest.approx(400)
    assert row["gap_to_lowest_reference_mev_per_O2"] == pytest.approx(420)
    assert row["reference_status"] == "insufficient_references"
    assert report["candidate_policy"] == "preserve_and_review"
    assert row["geometry"]["status"] == "unavailable"
    assert state == before
    json.dumps(report, allow_nan=False)


def test_energy_group_can_support_hull_when_single_influence_is_small():
    state = sample()
    add(state, "near", -0.399)
    row = row_for(diagnose_phase_diagram(state))
    assert row["group_size"] == 2
    assert row["priority_mev_per_O2"] == pytest.approx(1)
    assert row["group_priority_mev_per_O2"] == pytest.approx(400)


def test_bounded_agent_context_shows_one_representative_per_group():
    state = sample()
    add(state, "near", -0.399)
    summary = quality_summary(state)
    groups = [r["group_id"] for r in summary["priority_structures"]]
    assert len(groups) == len(set(groups))
    assert any("r1" in r["record_ids"] for r in summary["priority_structures"])


def test_explicit_two_mev_policy_not_chained_or_cross_phase():
    state = sample()
    add(state, "edge", -0.398)
    add(state, "chain", -0.396)
    add(state, "other-phase", -0.3999, phase="O1")
    report = diagnose_phase_diagram(state, policy={"group_span_mev_per_O2": 2})
    row = row_for(report)
    assert row["group_size"] == 2
    assert row["group_priority_mev_per_O2"] == pytest.approx(0.1)
    assert row["gap_to_lowest_reference_mev_per_O2"] == pytest.approx(4)


def test_reference_window_fifty_from_other_group_minimum_and_no_high_energy_bias():
    state = sample()
    for name, e in [("p1", 0.03), ("p2", 0.07), ("p3", 0.070001), ("high", 10)]:
        add(state, name, e)
    add(state, "cross-phase", -0.1, phase="O1")
    row = row_for(diagnose_phase_diagram(state))
    assert row["reference_count"] == 3
    assert row["gap_to_reference_median_mev_per_O2"] == pytest.approx(430)


def test_duplicate_geometry_is_removed_together_and_does_not_count_twice():
    state = sample()
    duplicate = deepcopy(state["phase_diagrams"]["mlip"]["entries"][1])
    duplicate["record_id"] = "duplicate"
    state["phase_diagrams"]["mlip"]["entries"].append(duplicate)
    report = diagnose_phase_diagram(state)
    assert report["coverage"]["unique_geometries"] == 4
    assert row_for(report)["hull_lift_mev_per_O2"] == pytest.approx(400)
    assert row_for(report)["record_ids"] == ["duplicate", "r1"]


@pytest.mark.parametrize("energy,review", [(-0.049, False), (-0.050, False), (-0.051, True)])
def test_twenty_mev_review_trigger_and_no_geometry_for_small_influence(energy, review):
    row = row_for(diagnose_phase_diagram(sample((0, energy, 0.0, 0))))
    assert row["review_recommended"] is review
    assert (row["geometry"]["status"] != "not_triggered") is review


def test_endpoint_without_alternative_is_unknown_not_zero():
    row = row_for(diagnose_phase_diagram(sample()), "r0")
    assert row["priority_mev_per_O2"] is None and row["review_recommended"]


def test_endpoint_with_alternative_uses_fixed_reference_not_rebased_zero():
    state = sample()
    add(state, "endpoint-alt", 0.7, x=0)
    assert row_for(diagnose_phase_diagram(state), "r0")["priority_mev_per_O2"] == pytest.approx(700)


def test_affine_energy_reference_does_not_change_influence_or_same_composition_gaps():
    state = sample()
    base = row_for(diagnose_phase_diagram(state))
    for row in state["phase_diagrams"]["mlip"]["entries"]:
        row["eform_per_O2"] += 3 + 2 * row["composition"]["Na"]
    row = row_for(diagnose_phase_diagram(state))
    assert row["priority_mev_per_O2"] == pytest.approx(base["priority_mev_per_O2"])
    assert row["gap_to_lowest_reference_mev_per_O2"] == pytest.approx(
        base["gap_to_lowest_reference_mev_per_O2"]
    )


def test_wrong_model_and_invalid_inputs_fail_explicitly():
    state = sample()
    state["active_model_version"] = "m2"
    assert quality_summary(state)["status"] == "diagnostic_unavailable"
    with pytest.raises(ValueError):
        diagnose_phase_diagram(sample(), policy={"hull_shift_mev_per_O2": float("nan")})


def test_report_cache_is_reused_and_bound_to_geometry_file_changes(tmp_path):
    state = sample()
    snapshot = state["phase_diagrams"]["mlip"]
    snapshot["csv_path"] = str(tmp_path / "phase_diagram.csv")
    first = get_phase_quality_report(state, write=True)
    path = Path(first["report_path"])
    timestamp = path.stat().st_mtime_ns
    assert get_phase_quality_report(state, write=True) == first
    assert path.stat().st_mtime_ns == timestamp
    state["phase_diagrams"]["mlip"]["entries"][1]["eform_per_O2"] = -0.5
    second = get_phase_quality_report(state, write=True)
    assert second["report_path"] != first["report_path"]


def test_candidate_signal_exact_binding_preserves_qbc_ehull_order_and_count():
    from phase_agent.science.qbc.build_candidate_metrics import build_qbc_candidate_metrics

    state = sample()
    candidates = [
        {
            "candidate_id": "c1",
            "structure_sha256": "g1",
            "model_version": "m1",
            "predicted_Ehull": 0,
            "qbc": {"energy_per_atom_std": 0.04},
        },
        {
            "candidate_id": "c2",
            "structure_id": "s1",
            "structure_sha256": "other",
            "model_version": "m1",
        },
        {"candidate_id": "c3", "structure_sha256": "g1", "model_version": "m0"},
    ]
    before = deepcopy(candidates)
    result = candidate_quality_evidence(state, candidates)
    assert candidates == before
    assert len(result) == 3 and [r["candidate_id"] for r in result] == ["c1", "c2", "c3"]
    assert result[0]["phase_quality_signal"]["priority_mev_per_O2"] == pytest.approx(400)
    assert result[0]["qbc"] == candidates[0]["qbc"] and result[0]["predicted_Ehull"] == 0
    assert all(r["phase_quality_signal"]["status"] == "unavailable" for r in result[1:])
    metrics = build_qbc_candidate_metrics(result)
    assert not metrics["decision_made"]
    assert metrics["metrics"][0]["phase_quality_signal"]["priority_mev_per_O2"] == pytest.approx(
        400
    )


def test_geometry_change_is_observation_not_error_or_score_boost(monkeypatch):
    import phase_agent.analysis.phase.phase_quality_geometry as geometry

    monkeypatch.setattr(
        geometry, "inspect_geometry", lambda row: {"non_six_fraction": 1, "volume_per_O2_A3": 25}
    )
    report = diagnose_phase_diagram(sample())
    row = row_for(report)
    assert row["priority_mev_per_O2"] == pytest.approx(400)
    assert "相变或重构" in row["geometry"]["interpretation"]
    assert "initial_unavailable" in row["geometry"]


def test_decision_context_exposes_joint_evidence():
    from phase_agent.analysis.state.build_decision_context import build_decision_context
    from phase_agent.graphs.dialogue.evidence import conversation_facts

    state = sample()
    context = build_decision_context(state)
    quality = context["current_phase_diagram"]["mlip"]["quality_diagnostics"]
    assert quality["decision_role"] == "supporting_evidence_alongside_ehull_and_qbc"
    assert conversation_facts(state, {})["phase_diagram_quality"]["status"] == "review_required"
