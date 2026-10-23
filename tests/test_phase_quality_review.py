from copy import deepcopy

import pytest

from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_review import apply_numbered_exclusions
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from tests.test_phase_quality import sample


def fixture():
    state = sample()
    manager = PhaseDataManager({"P": ["O3"], "H": {"max_det": 100}, "TM_ratio": {"Co": 1}})
    bid = manager.add_branch(P="O3", H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], x=0.5, T=["Co"])
    sid = manager.add_structure(branch_id=bid, arrangement={"positions": [1]})
    state["phase_diagrams"]["mlip"]["entries"][1]["structure_id"] = sid
    state["tasks"] = [{"task_id": "t1", "relax_mc_bundle_id": "b1", "model_version": "m1"}]
    state["relax_mc_batches"] = {
        "b1": {"status": "completed", "model_version": "m1", "import_checksum": "verified"}
    }
    state["phase_records"] = [
        {
            **r,
            "energy_method": "mlip",
            "energy": r["eform_per_O2"],
            "energy_unit": "eV",
            "model_version": "m1",
            "status": "completed",
            "source_task_id": "t1" if r["record_id"] == "r1" else "other",
        }
        for r in state["phase_diagrams"]["mlip"]["entries"]
    ]
    report = diagnose_phase_diagram(state)
    report.update(bundle_id="b1", import_checksum="verified")
    from phase_agent.analysis.phase.phase_quality_markdown import review_groups

    number = next(
        g["number"]
        for g in review_groups(report)
        if any("r1" in r["record_ids"] for r in g["members"])
    )
    return state, manager, report, number, bid, sid


def test_explicit_exclusion_only_this_round_and_branch_membership_preserves_history():
    state, manager, report, number, bid, sid = fixture()
    before = deepcopy(state)
    result = apply_numbered_exclusions(
        state,
        manager,
        report,
        bundle_id="b1",
        numbers=[number],
        reason="人工核查",
        human_requested=True,
    )
    assert result["excluded_record_ids"] == ["r1"]
    assert state == before and not result["submitted"]
    assert sid not in manager.data["branches"][bid]["structure_ids"]
    assert sid not in manager.data["structures"]
    assert sid in manager.data["phase_quality_archived_structures"]
    assert len(result["state"]["phase_records"]) == 3
    assert (
        result["state"]["phase_quality_archived_records"][0]["status"]
        == "excluded_phase_quality_review"
    )
    from phase_agent.analysis.phase.refresh_identified_phases import refresh_identified_phases

    refreshed = deepcopy(result["state"])
    refresh_identified_phases(refreshed)
    assert (
        refreshed["phase_quality_archived_records"][0]["status"] == "excluded_phase_quality_review"
    )
    assert "r1" not in {
        r["record_id"] for r in result["state"]["phase_diagrams"]["mlip"]["entries"]
    }
    assert manager.data["phase_quality_review_log"][0]["reason"] == "人工核查"
    # Later registration is allowed and does not erase past stage history.
    assert manager.add_structure(branch_id=bid, arrangement={"positions": [1]}) == sid
    assert sid in manager.data["branches"][bid]["structure_ids"]
    new_records = [
        {
            **r,
            "record_id": "new-" + r["record_id"],
            "model_version": "m2",
            "source_version": "m2",
            "status": "completed",
        }
        for r in before["phase_records"]
    ]
    output = update_phase_diagram(
        [*result["state"]["phase_records"], *new_records],
        active_model_version="m2",
        output_state=result["state"],
    )
    assert output["diagrams"]["mlip"]["status"] == "completed"
    assert "new-r1" in {r["record_id"] for r in output["diagrams"]["mlip"]["entries"]}


@pytest.mark.parametrize(
    "failure",
    ["no_human", "stale_hull", "old_model", "other_batch", "invalid_number", "historical_record"],
)
def test_review_rejects_unbound_or_unrequested_changes_without_mutation(failure):
    state, manager, report, number, *_ = fixture()
    human = True
    if failure == "no_human":
        human = False
    elif failure == "stale_hull":
        report["hull_version"] = "old"
    elif failure == "old_model":
        report["model_version"] = "old"
    elif failure == "other_batch":
        report["bundle_id"] = "b2"
    elif failure == "invalid_number":
        number = 999
    else:
        state["phase_records"][1]["source_task_id"] = "historical-task"
    before = deepcopy(manager.data)
    with pytest.raises(ValueError):
        apply_numbered_exclusions(
            state,
            manager,
            report,
            bundle_id="b1",
            numbers=[number],
            reason="审查",
            human_requested=human,
        )
    assert manager.data == before
