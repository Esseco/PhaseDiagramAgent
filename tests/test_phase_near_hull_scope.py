from copy import deepcopy

from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_reports import candidate_quality_evidence
from phase_agent.science.qbc.build_candidate_metrics import build_qbc_candidate_metrics
from tests.test_phase_quality import sample, add, row_for


def test_far_D_does_not_raise_quality_sampling_priority_even_if_very_large():
    state = sample()
    add(state, "far", 10, x=0.25)
    for i in range(6):
        add(state, f"far-reference-{i}", 11 + 0.001 * i, x=0.25)
    report = diagnose_phase_diagram(state)
    row = row_for(report, "far")
    assert row["D_mev_per_O2"] > 900
    assert not row["near_current_hull"] and not row["quality_sampling_evidence_eligible"]
    assert row["dft_evidence_priority_mev_per_O2"] == 0
    assert not row["preview_excluded"] and row["tier"] == 0
    candidate = {
        "candidate_id": "c",
        "structure_sha256": "far",
        "predicted_Ehull": 2.0,
        "ehull_unit": "eV/atom",
        "phase_diagram_version": "h1",
    }
    enriched = candidate_quality_evidence(state, [candidate], report=report)[0]
    assert enriched["predicted_Ehull"] == 2.0
    assert enriched["phase_quality_signal"]["dft_evidence_priority_mev_per_O2"] == 0


def test_near_prefix_precedes_supported_far_prefix_and_hidden_point_waits_for_recursion():
    state = sample((0, -0.5, -0.4, 0))
    for i in range(6):
        add(state, f"ref-{i}", -0.1 + 0.001 * i)
    report = diagnose_phase_diagram(state)
    first = report["analysis_layers"][0]
    island = next(c for c in report["energy_islands"] if "r1" in c["record_ids"])
    assert island["record_ids"] == ["r1"]
    hidden = row_for(report, "r2")
    assert hidden["geometry_key"] not in first["joint_geometry_keys"]
    assert not hidden["near_current_hull"] and hidden["conditional_only"]
    assert hidden["sampling_evidence_scope"] == "conditional_near_hull"
    assert hidden["preview_excluded"] and not hidden["automatic_exclusion_eligible"]
    assert any(
        e["distance_to_conditional_hull_mev_per_O2"] <= 50 for e in hidden["conditional_evidence"]
    )


def test_stale_quality_hull_cannot_attach_to_current_candidate_and_metrics_keep_units():
    state = sample()
    report = diagnose_phase_diagram(state)
    candidate = {
        "candidate_id": "c",
        "structure_sha256": "g1",
        "predicted_Ehull": 0.02,
        "phase_diagram_version": "h1",
        "ehull_unit": "eV/atom",
    }
    stale = deepcopy(report)
    stale["hull_version"] = "old-hull"
    result = candidate_quality_evidence(state, [candidate], report=stale)
    assert result[0]["phase_quality_signal"]["status"] == "unavailable"
    metrics = build_qbc_candidate_metrics(result)["metrics"][0]
    assert metrics["predicted_Ehull"] == 0.02
    assert metrics["phase_diagram_version"] == "h1" and metrics["ehull_unit"] == "eV/atom"


def test_DFT_candidate_reads_new_formal_Ehull_after_executed_removal(tmp_path):
    from pymatgen.core import Structure, Lattice
    from tests.test_phase_quality_review import fixture
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    from phase_agent.analysis.phase.phase_quality_actions import apply_quality_review
    from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report
    from phase_agent.science.qbc.post_mc_candidates import post_mc_candidates

    state, manager, *_ = fixture()
    path = tmp_path / "recovered.vasp"
    Structure(
        Lattice.cubic(8),
        ["Na", "Co", "Co", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.2, 0.2, 0.2],
            [0.7, 0.7, 0.7],
            [0.1, 0.1, 0.4],
            [0.3, 0.3, 0.4],
            [0.6, 0.6, 0.9],
            [0.8, 0.8, 0.9],
        ],
    ).to(filename=path, fmt="poscar")
    state["phase_records"][2]["structure_path"] = str(path)
    state["tasks"].append(
        {
            "task_id": "mc-r2",
            "stage": "deep_search",
            "status": "completed",
            "model_version": "m1",
            "structure_id": "s2",
            "branch_id": "b2",
            "outputs": {"structure_path": str(path)},
        }
    )
    state["phase_diagrams"].update(
        update_phase_diagram(state["phase_records"], active_model_version="m1", output_state=state)[
            "diagrams"
        ]
    )
    original = post_mc_candidates(state, "m1")[0]
    report = get_phase_quality_report(state)
    changed = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row_for(report)["review_id"]],
        decision="auto_remove",
        reason="test-current-hull",
        backup_root=tmp_path / "backup",
    )["state"]
    current = post_mc_candidates(changed, "m1")[0]
    assert current["phase_diagram_version"] == changed["phase_diagrams"]["mlip"]["version"]
    assert current["phase_diagram_version"] != original["phase_diagram_version"]
    assert current["predicted_Ehull"] < original["predicted_Ehull"]
    assert current["ehull_unit"] == "eV/atom"
