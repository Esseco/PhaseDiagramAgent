from copy import deepcopy
from hashlib import sha256

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.analysis.phase.phase_framework import inspect_framework, periodic_framework
from phase_agent.analysis.phase.phase_quality import DEFAULT_POLICY, diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_actions import (
    apply_quality_review,
    automatic_round_review,
)
from phase_agent.analysis.phase.phase_quality_reports import (
    get_phase_quality_report,
    candidate_quality_evidence,
)
from tests.test_phase_quality import sample, add, row_for
from tests.test_phase_quality_review import fixture


def slab():
    return Structure(
        Lattice.hexagonal(2.8, 10),
        ["Co", "O", "O"],
        [[0, 0, 0.5], [1 / 3, 2 / 3, 0.42], [2 / 3, 1 / 3, 0.58]],
    )


def evidence(tmp_path, initial, final):
    before, after = tmp_path / "before.vasp", tmp_path / "after.vasp"
    initial.to(filename=str(before), fmt="poscar")
    final.to(filename=str(after), fmt="poscar")
    row = {
        "structure_path": str(after),
        "structure_sha256": sha256(after.read_bytes()).hexdigest(),
        "composition": final.composition.as_dict(),
    }
    return inspect_framework(row, before), row, before


def test_intact_rotated_and_translated_cell_keeps_framework(tmp_path):
    initial = slab()
    final = initial.copy()
    final.translate_sites(range(len(final)), [0.3, 0.2, 0.4])
    # Swap Cartesian axes without changing periodic topology.
    import numpy as np

    final = Structure(
        Lattice(np.asarray(final.lattice.matrix)[:, [2, 0, 1]]), final.species, final.frac_coords
    )
    result, _, _ = evidence(tmp_path, initial, final)
    assert result["status"] == "no_clear_destruction"
    assert not result["auto_confirmed"]
    import json

    assert json.loads(json.dumps(result)) == result


def test_collapse_has_robust_three_dimensional_bond_network(tmp_path):
    initial = slab()
    final = Structure(Lattice.hexagonal(2.8, 1), initial.species, initial.frac_coords)
    result, _, _ = evidence(tmp_path, initial, final)
    assert result["auto_confirmed"]
    assert all(p["tm_rank_fractions"]["3"] == 1 for p in result["final"])


def test_partial_connected_fragment_is_sufficient_not_majority(tmp_path):
    initial = slab()
    initial.make_supercell([2, 1, 4])
    final = initial.copy()
    # Separate the O neighbors of just one slab, leaving three slabs intact.
    centers = sorted({round(float(s.frac_coords[2]), 5) for s in final if s.specie.symbol == "Co"})
    z = centers[0]
    for i, site in enumerate(final):
        if site.specie.symbol == "O" and abs(site.frac_coords[2] - z) < 0.03:
            coords = site.frac_coords.copy()
            coords[2] = z + 0.12
            final.replace(i, site.specie, coords)
    result, _, _ = evidence(tmp_path, initial, final)
    assert result["auto_confirmed"]
    assert 0 < result["affected_tm_fraction"] < 0.5
    assert result["affected_tm_count"] >= 2


def test_bad_file_binding_and_missing_initial_are_uncertain(tmp_path):
    initial = slab()
    result, row, before = evidence(tmp_path, initial, initial)
    row["structure_sha256"] = "0" * 64
    assert inspect_framework(row, before)["status"] == "uncertain"
    assert inspect_framework(row, None)["status"] == "uncertain"


def test_cutoff_twenty_group_span_ten_and_three_recursive_steps():
    assert DEFAULT_POLICY == {
        "group_span_mev_per_O2": 10.0,
        "reference_window_mev_per_O2": 50.0,
        "hull_shift_mev_per_O2": 50.0,
        "outlier_gap_mev_per_O2": 50.0,
        "direct_removal_mev_per_O2": 100.0,
        "recursive_max_layers": None,
        "joint_max_order": 5,
    }
    for energy, high in ((-0.049, False), (-0.050, False), (-0.051, True)):
        assert (
            row_for(diagnose_phase_diagram(sample((0, energy, 0.0, 0))))["review_recommended"]
            is high
        )
    state = sample()
    add(state, "edge", -0.390)
    add(state, "chain", -0.380)
    row = row_for(diagnose_phase_diagram(state))
    assert row["group_size"] == 2
    assert row["gap_to_lowest_reference_mev_per_O2"] == pytest.approx(20)


def test_retain_preserves_metrics_continues_and_binds_evidence(tmp_path):
    state, manager, _, number, *_ = fixture()
    state["phase_diagrams"]["mlip"]["entries"][1]["eform_per_O2"] = -0.08
    state["phase_records"][1].update(eform_per_O2=-0.08, energy=-0.08)
    report = get_phase_quality_report(state)
    row = row_for(report)
    result = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row["review_id"]],
        decision="retain",
        reason="先用DFT验证",
        human_requested=True,
    )
    assert result["continue_dft_selection"]
    later = get_phase_quality_report(result["state"])
    retained = row_for(later)
    assert retained["disposition"] == "human_retained"
    assert not retained["review_pending"]
    assert retained["priority_mev_per_O2"] == row["priority_mev_per_O2"]
    changed = deepcopy(result["state"])
    changed["phase_diagrams"]["mlip"]["version"] = "changed"
    assert row_for(get_phase_quality_report(changed))["disposition"] == "pending_human"
    with pytest.raises(ValueError, match="改变"):
        apply_quality_review(
            changed,
            manager,
            report,
            review_ids=[row["review_id"]],
            decision="remove",
            reason="删除",
            backup_root=tmp_path,
            human_requested=True,
        )


def test_human_remove_backed_up_exact_result_and_filters_candidates(tmp_path):
    state, manager, _, _, bid, sid = fixture()
    manager.data["structures"][sid]["stage_history"]["relax_and_feature"] = [
        {"status": "completed"}
    ]
    assert manager.coverage()
    state["phase_records"][1]["structure_path"] = str(tmp_path / "source.vasp")
    Path = type(tmp_path)
    Path(state["phase_records"][1]["structure_path"]).write_text("raw result", encoding="utf-8")
    report = get_phase_quality_report(state)
    result = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row_for(report)["review_id"]],
        decision="remove",
        reason="用户确认破坏",
        backup_root=tmp_path / "backup",
        human_requested=True,
    )
    assert result["excluded_record_ids"] == ["r1"]
    assert sid not in manager.data["branches"][bid]["structure_ids"]
    assert not manager.coverage() and not result["state"]["coverage"]
    assert list((tmp_path / "backup").rglob("audit.json"))
    candidates = [
        {"structure_path": state["phase_records"][1]["structure_path"]},
        {"candidate_id": "other", "model_version": "m1"},
    ]
    assert len(candidate_quality_evidence(result["state"], candidates)) == 1


def test_uncertain_or_conditional_cannot_be_auto_removed(tmp_path):
    state, manager, *_ = fixture()
    state["phase_diagrams"]["mlip"]["entries"][1]["eform_per_O2"] = -0.08
    state["phase_records"][1].update(eform_per_O2=-0.08, energy=-0.08)
    report = get_phase_quality_report(state)
    with pytest.raises(ValueError, match="严格"):
        apply_quality_review(
            state,
            manager,
            report,
            review_ids=[row_for(report)["review_id"]],
            decision="auto_remove",
            reason="不能误删",
            backup_root=tmp_path,
        )


def test_framework_graph_rank_invariant_under_supercell():
    initial = slab()
    assert periodic_framework(initial, 2.2)["tm_rank_fractions"]["2"] == 1
    initial.make_supercell([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    assert periodic_framework(initial, 2.2)["tm_rank_fractions"]["2"] == 1


def test_real_framework_rule_drives_automatic_round_exclusion_and_is_idempotent(tmp_path):
    state, manager, _, _, bid, sid = fixture()
    initial = slab()
    initial.make_supercell([2, 1, 1])
    initial.append("Na", [0.1, 0.1, 0])
    final = Structure(
        Lattice([initial.lattice.matrix[0], initial.lattice.matrix[1], [0, 0, 1]]),
        initial.species,
        initial.frac_coords,
    )
    framework, row, before = evidence(tmp_path, initial, final)
    assert framework["auto_confirmed"]
    for target in (state["phase_records"][1], state["phase_diagrams"]["mlip"]["entries"][1]):
        target.update(row, source_task_id="t1")
    state["tasks"][0].update(structure_path=str(before), status="completed")
    state["relax_mc_batches"]["b1"]["result_summary"] = {
        "status": "completed",
        "model_version": "m1",
    }
    state["phase_diagrams"]["mlip"]["csv_path"] = str(tmp_path / "current.csv")
    report = get_phase_quality_report(state, write=True)
    assert get_phase_quality_report(state)["structures"] == report["structures"]
    assert row_for(report)["automatic_exclusion_eligible"]
    result = automatic_round_review(
        state, manager, bundle_id="b1", backup_root=tmp_path / "backups"
    )
    assert result["excluded_record_ids"] == ["r1"]
    assert sid not in manager.data["branches"][bid]["structure_ids"]
    again = automatic_round_review(
        result["state"], manager, bundle_id="b1", backup_root=tmp_path / "backups"
    )
    assert again["status"] == "already_reviewed"


def test_durable_human_retain_invalidates_stale_writer(tmp_path):
    import json
    from phase_agent.analysis.phase.phase_quality_persistence import commit_human_quality_review
    from phase_agent.tools.step_runner.file_protocol import write_json

    state, manager, *_ = fixture()
    state_path, ledger_path = tmp_path / "state.json", tmp_path / "phase_data.json"
    manager.save(ledger_path)
    write_json(state_path, state)
    state["phase_diagrams"]["mlip"]["entries"][1]["eform_per_O2"] = -0.08
    state["phase_records"][1].update(eform_per_O2=-0.08, energy=-0.08)
    write_json(state_path, state)
    report = get_phase_quality_report(state)
    result = commit_human_quality_review(
        state_path,
        state,
        manager,
        review_ids=[row_for(report)["review_id"]],
        decision="retain",
        user_message="这个先不管，继续选点",
        ledger_path=ledger_path,
    )
    assert result["continue_dft_selection"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["phase_quality_decisions"]
    with pytest.raises(ValueError, match="stale"):
        write_json(state_path, state)


def test_semantic_quality_review_contract_is_explicit_and_typed():
    from phase_agent.decisions.agent.dialogue_contract import dialogue_errors, is_dialogue

    action = {
        "kind": "phase_quality_review",
        "decision": "retain",
        "review_ids": ["pq-" + "a" * 24],
    }
    assert is_dialogue(action)
    assert not dialogue_errors(action, True)
    assert dialogue_errors({**action, "decision": "guess"}, True)


def test_joint_requires_every_geometry_confirmed_and_cannot_be_split(monkeypatch, tmp_path):
    import phase_agent.analysis.phase.phase_quality as quality

    state, manager, *_ = fixture()
    add(state, "mask", -0.399, phase="P3")
    # Keep this a pure joint-H test: no same-phase high-D reference for r1.
    state["phase_diagrams"]["mlip"]["entries"][2]["phase"] = "P2"
    state["phase_records"][2]["phase"] = "P2"

    # A synthetic independent geometry inspection isolates the removal-set gate.
    def inspected(point, state, cache):
        return {
            "status": "observed",
            "framework": {"status": "destroyed_confirmed", "auto_confirmed": True},
        }

    monkeypatch.setattr(quality, "_cached_geometry", inspected)
    report = get_phase_quality_report(state)
    selected = row_for(report)
    assert selected["hull_lift_mev_per_O2"] < 20
    assert selected["automatic_exclusion_eligible"]
    with pytest.raises(ValueError, match="拆开"):
        apply_quality_review(
            state,
            manager,
            report,
            review_ids=[selected["review_id"]],
            decision="auto_remove",
            reason="不能拆开",
            backup_root=tmp_path,
        )

    def uncertain_mask(point, state, cache):
        return (
            {"status": "observed", "framework": {"status": "uncertain", "auto_confirmed": False}}
            if point["phase"] == "P3"
            else inspected(point, state, cache)
        )

    monkeypatch.setattr(quality, "_cached_geometry", uncertain_mask)
    assert not row_for(get_phase_quality_report(state))["automatic_exclusion_eligible"]


def test_boundary_cleanup_updates_all_methods_but_keeps_backed_up_history(tmp_path):
    from phase_agent.analysis.phase.phase_boundary_cleanup import cleanup_phase_boundary
    from tests.test_phase_recursive import CO_RULE

    state, manager, *_ = fixture()
    state["confirmed_config"] = {"system": {"boundary": {"P": CO_RULE}}}
    manager.data["structures"][state["phase_records"][1]["structure_id"]]["composition"] = {
        "Co": 1,
        "Na": 0.5,
        "O": 2,
    }
    bad_bid = manager.add_branch(P="O1", H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], x=0.5, T=["Co"])
    bad_sid = manager.add_structure(
        branch_id=bad_bid, arrangement={"a": 2}, composition={"Co": 1, "Na": 0.5, "O": 2}
    )
    invalid = {
        **deepcopy(state["phase_records"][1]),
        "record_id": "invalid-dft",
        "phase": "O1",
        "structure_id": bad_sid,
        "energy_method": "dft",
    }
    # Supply a valid Na0 O1 endpoint alongside the invalid historical O3 endpoint.
    endpoint = {**deepcopy(state["phase_records"][0]), "record_id": "valid-na0", "phase": "O1"}
    state["phase_records"].extend([invalid, endpoint])
    result = cleanup_phase_boundary(state, manager, backup_root=tmp_path, human_requested=True)
    assert {"r0", "invalid-dft"} <= set(result["excluded_record_ids"])
    assert not manager.data["branches"][bad_bid]["structure_ids"]
    assert bad_sid in manager.data["structures"]
    assert all(
        r["phase"] == "O1"
        for r in result["state"]["phase_diagrams"]["mlip"]["entries"]
        if r["composition"].get("Na", 0) == 0
    )
    assert list(tmp_path.rglob("audit.json"))
    before_repeat = deepcopy(manager.data)
    repeated = cleanup_phase_boundary(
        result["state"], manager, backup_root=tmp_path, human_requested=True
    )
    assert repeated["status"] == "already_clean"
    assert not repeated["excluded_record_ids"] and manager.data == before_repeat


def test_recovery_cannot_restore_excluded_result_or_boundary_outlier_to_pool(tmp_path):
    from phase_agent.analysis.phase.update_local_mlip_hull_pool import update_local_mlip_hull_pool
    from tests.test_versioned_mlip_hulls import task
    from tests.test_phase_recursive import CO_RULE

    geometry = tmp_path / "raw.vasp"
    geometry.write_text("raw calculation remains", encoding="utf-8")
    valid = task(geometry, "valid", {"Na": 1, "Co": 1, "O": 2}, -3)
    broken = task(geometry, "broken", {"Na": 0.5, "Co": 1, "O": 2}, -20)
    outside = task(geometry, "outside", {"Co": 1, "O": 2}, -30)
    state = {
        "active_model_version": "m1",
        "tasks": [valid, broken, outside],
        "confirmed_config": {"system": {"boundary": {"P": CO_RULE}}},
        "phase_records": [
            {
                "record_id": "excluded",
                "source_task_id": "broken",
                "structure_path": str(geometry),
                "status": "excluded_phase_quality_review",
                "model_version": "m1",
            }
        ],
    }
    result = update_local_mlip_hull_pool(
        state,
        config={"system": {"system_id": "Co"}, "mlip": {"version": "m1"}},
        path=tmp_path / "pool.json",
    )
    pool = result["branch_hull_batches"][result["current_branch_hull_version"]]
    assert [r["source_task_id"] for r in pool["records"]] == ["valid"]
