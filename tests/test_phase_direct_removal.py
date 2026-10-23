from copy import deepcopy

import pytest

from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_actions import apply_quality_review
from phase_agent.analysis.phase.phase_quality_reports import (
    get_phase_quality_report,
    review_markdown,
)
from tests.test_phase_quality import sample, add, row_for
from tests.test_phase_quality_review import fixture


@pytest.mark.parametrize(
    "lift,eligible", [(99.999, False), (100, False), (100.001, True), (667.45, True)]
)
def test_strict_single_cutoff_without_framework(lift, eligible):
    report = diagnose_phase_diagram(sample((0, -lift / 1000, 0.05, 0)))
    row = row_for(report)
    assert row["automatic_exclusion_eligible"] is eligible
    if eligible:
        assert row["automatic_exclusion_reason"] == "single_impact_above_100"
        assert row["geometry"]["status"] == "unavailable"


def test_group_and_conditional_lifts_cannot_substitute_for_current_single_lift():
    state = sample()
    add(state, "mask", -0.399)
    row = row_for(diagnose_phase_diagram(state))
    assert row["group_hull_lift_mev_per_O2"] > 100
    assert row["hull_lift_mev_per_O2"] == pytest.approx(1)
    assert not row["automatic_exclusion_eligible"]
    state = sample()
    add(state, "hidden", -0.2)
    hidden = row_for(diagnose_phase_diagram(state), "hidden")
    assert any(e["D_mev_per_O2"] > 100 for e in hidden["conditional_evidence"])
    assert hidden["conditional_only"]  # A far backup only activates after recursive exposure.
    assert hidden["hull_lift_mev_per_O2"] == 0
    assert not hidden["automatic_exclusion_eligible"]


def test_direct_exclusion_backs_up_and_documents_reason_value_and_path(tmp_path):
    state, manager, *_ = fixture()
    state["phase_diagrams"]["mlip"]["entries"][1]["structure_path"] = str(tmp_path / "missing.vasp")
    state["phase_records"][1]["structure_path"] = str(tmp_path / "missing.vasp")
    report = get_phase_quality_report(state)
    before = deepcopy(state)
    result = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row_for(report)["review_id"]],
        decision="auto_remove",
        reason="当前单结构影响>100，按策略移除",
        backup_root=tmp_path / "backups",
    )
    assert state == before
    assert result["excluded_record_ids"] == ["r1"]
    after = get_phase_quality_report(result["state"])
    markdown = review_markdown(after, bundle_id="test")
    assert "400.000" in markdown and "missing.vasp" in markdown
    assert "不代表已确认结构破坏" in markdown
    assert list((tmp_path / "backups").rglob("audit.json"))


def test_human_restore_reinstates_hull_and_does_not_immediately_redelete(tmp_path):
    state, manager, *_ = fixture()
    report = get_phase_quality_report(state)
    review_id = row_for(report)["review_id"]
    removed = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[review_id],
        decision="auto_remove",
        reason="单结构>100",
        backup_root=tmp_path / "backup",
    )["state"]
    restored = apply_quality_review(
        removed,
        manager,
        None,
        review_ids=[review_id],
        decision="restore",
        reason="人工确认保留该结果",
        human_requested=True,
        backup_root=tmp_path / "backup",
    )
    assert restored["restored_record_ids"] == ["r1"]
    assert any(
        r["record_id"] == "r1" for r in restored["state"]["phase_diagrams"]["mlip"]["entries"]
    )
    row = row_for(get_phase_quality_report(restored["state"]))
    assert row["hull_lift_mev_per_O2"] > 100
    assert row["disposition"] == "human_restored" and not row["automatic_exclusion_eligible"]
    assert restored["state"]["phase_quality_exclusions"]
    assert "已人工恢复记录" in review_markdown(
        get_phase_quality_report(restored["state"]), bundle_id="test"
    )


def test_restore_requires_human_current_model_and_boundary(tmp_path):
    state, manager, *_ = fixture()
    report = get_phase_quality_report(state)
    review_id = row_for(report)["review_id"]
    removed = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[review_id],
        decision="auto_remove",
        reason="单结构>100",
        backup_root=tmp_path / "backup",
    )["state"]
    for problem in ("human", "model", "boundary"):
        current = deepcopy(removed)
        if problem == "model":
            current["active_model_version"] = "m2"
        if problem == "boundary":
            current["confirmed_config"] = {
                "system": {
                    "boundary": {"P": {"at_x": {"0": ["O1"], "1": ["O1"]}, "intermediate": ["O1"]}}
                }
            }
        with pytest.raises(ValueError):
            apply_quality_review(
                current,
                manager,
                None,
                review_ids=[review_id],
                decision="restore",
                reason="恢复",
                human_requested=problem != "human",
                backup_root=tmp_path / "backup",
            )


@pytest.mark.parametrize("count", [4, 5])
def test_masked_four_and_five_way_geometry_and_group_combinations(count):
    state = sample((0, -0.2, 0.05, 0))
    for i, phase in enumerate(["P2", "P3", "OP2", "O1"][: count - 1]):
        add(state, "mask-" + phase, -0.199 + i * 0.001, phase=phase)
    report = diagnose_phase_diagram(state)
    for mode in ("geometry", "energy_group"):
        cases = report["analysis_layers"][0]["joint_cases"]["cases"]
        case = next(c for c in cases if c["mode"] == mode and c["order"] == count)
        assert case["hull_lift_mev_per_O2"] == pytest.approx(200)
    assert not row_for(report)["automatic_exclusion_eligible"]


@pytest.mark.parametrize("seed", [2, 7, 11])
def test_five_way_pruning_matches_exhaustive(seed):
    from itertools import combinations
    import random
    from phase_agent.analysis.phase.phase_hull_sensitivity import joint_cases
    from phase_agent.analysis.phase.phase_quality import _hull, _influence

    rng = random.Random(seed)
    points = [
        {
            "key": str(i),
            "x": x,
            "energy_mev_per_O2": rng.uniform(-100, 100),
            "rows": [{"record_id": str(i)}],
        }
        for i, x in enumerate([0, 0, 0.25, 0.5, 0.5, 0.75, 1, 1])
    ]
    hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in points])
    expected = set()
    for size in range(1, 6):
        for combo in combinations([p["key"] for p in points], size):
            keys = frozenset(combo)
            if any(previous < keys for previous in expected):
                continue
            lift = _influence(points, hull, set(keys))
            if lift is not None and lift >= 20:
                expected.add(keys)
    actual = joint_cases(points, {p["key"]: [p] for p in points}, threshold=20, order=5)
    assert {
        frozenset(c["unit_ids"]) for c in actual["cases"] if c["mode"] == "geometry"
    } == expected


def test_human_restore_persists_through_existing_control_transaction(tmp_path):
    import json
    from phase_agent.analysis.phase.phase_quality_persistence import commit_human_quality_review
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.decisions.agent.dialogue_contract import dialogue_errors

    state, manager, *_ = fixture()
    report = get_phase_quality_report(state)
    review_id = row_for(report)["review_id"]
    state = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[review_id],
        decision="auto_remove",
        reason="单结构>100",
        backup_root=tmp_path / "backup",
    )["state"]
    state_path, ledger = tmp_path / "state.json", tmp_path / "phase_data.json"
    write_json(state_path, state)
    manager.save(ledger)
    result = commit_human_quality_review(
        state_path,
        state,
        manager,
        review_ids=[review_id],
        decision="restore",
        user_message="恢复这个结构",
        ledger_path=ledger,
    )
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["phase_quality_restorations"]
    assert result["status"] == "restored"
    assert not dialogue_errors(
        {"kind": "phase_quality_review", "decision": "restore", "review_ids": [review_id]}, True
    )
