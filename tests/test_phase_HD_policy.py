from copy import deepcopy
import json

import pytest

from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_policy import (
    effective_policy,
    update_quality_policy,
    commit_quality_policy,
)
from phase_agent.analysis.phase.phase_quality_actions import apply_quality_review
from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report
from tests.test_phase_quality import sample, add, row_for
from tests.test_phase_quality_review import fixture


def test_manual_policy_precedence_audit_and_agent_adjustment():
    state = sample()
    manual = update_quality_policy(
        state, cutoffs={"h_cutoff": 60}, source="human", reason="人工阈值", expected_revision=0
    )["state"]
    assert effective_policy(manual)["hull_shift_mev_per_O2"] == 60
    with pytest.raises(ValueError, match="人工固定"):
        update_quality_policy(
            manual, cutoffs={"h_cutoff": 70}, source="agent", reason="修改", expected_revision=1
        )
    adjusted = update_quality_policy(
        manual, cutoffs={"d_cutoff": 80}, source="agent", reason="参考分布较宽", expected_revision=1
    )["state"]
    assert adjusted["phase_quality_policy"]["history"][-1]["source"] == "agent"
    assert effective_policy(adjusted)["outlier_gap_mev_per_O2"] == 80
    unlocked = update_quality_policy(
        adjusted,
        cutoffs={},
        release_manual=["h_cutoff"],
        source="human",
        reason="交给agent",
        expected_revision=2,
    )["state"]
    assert not unlocked["phase_quality_policy"]["manual_fields"]
    assert "phase_quality_policy" not in state


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_cutoff_cannot_change_state(value):
    state = sample()
    before = deepcopy(state)
    with pytest.raises(ValueError):
        update_quality_policy(
            state, cutoffs={"d_cutoff": value}, source="human", reason="test", expected_revision=0
        )
    assert state == before


def test_policy_change_is_persisted_without_deletion_or_science(tmp_path):
    state, manager, *_ = fixture()
    sp, lp = tmp_path / "state.json", tmp_path / "ledger.json"
    sp.write_text(json.dumps(state), encoding="utf-8")
    manager.save(lp)
    before = deepcopy(state["phase_records"])
    result = commit_quality_policy(
        sp,
        state,
        manager,
        ledger_path=lp,
        cutoffs={"d_cutoff": 75},
        source="human",
        reason="user",
        expected_revision=0,
    )
    saved = json.loads(sp.read_text(encoding="utf-8"))
    assert saved["phase_records"] == before and not result["submitted"]
    assert saved["phase_quality_policy"]["revision"] == 1
    with pytest.raises(ValueError):
        update_quality_policy(
            saved, cutoffs={"d_cutoff": 80}, source="agent", reason="stale", expected_revision=0
        )


def test_D_alone_can_trigger_existing_structure_verdict_without_large_H(monkeypatch):
    import phase_agent.analysis.phase.phase_quality as quality

    state = sample()
    add(state, "island", -0.16, x=0.25)
    for i in range(6):
        add(state, f"ref{i}", 0.25 + 0.001 * i, x=0.25)

    def verdict(point, state, cache):
        return {
            "status": "observed",
            "framework": {"status": "destroyed_confirmed", "auto_confirmed": True},
        }

    monkeypatch.setattr(quality, "_cached_geometry", verdict)
    report = diagnose_phase_diagram(state)
    row = row_for(report, "island")
    assert row["H_mev_per_O2"] == 0 and row["D_mev_per_O2"] > 400
    assert row["tier"] == 1 and row["automatic_exclusion_eligible"]


def test_conditional_high_H_or_D_never_authorizes_formal_deletion(monkeypatch):
    import phase_agent.analysis.phase.phase_quality as quality

    state = sample()
    add(state, "island", -0.1, x=0.25)
    for i in range(6):
        add(state, f"ref{i}", 0.25 + 0.001 * i, x=0.25)
    monkeypatch.setattr(
        quality,
        "_cached_geometry",
        lambda *args: {
            "status": "observed",
            "framework": {"auto_confirmed": True, "status": "destroyed_confirmed"},
        },
    )
    row = row_for(diagnose_phase_diagram(state), "island")
    assert row["H_mev_per_O2"] == 0 and not row["near_current_hull"]
    assert row["tier"] == 2 and row["conditional_only"]
    assert not row["automatic_exclusion_eligible"]


def test_archive_keeps_branch_peers_and_allows_new_registration_and_records(tmp_path):
    state, manager, _, _, bid, sid = fixture()
    peer = manager.add_structure(branch_id=bid, arrangement={"peer": 1})
    report = get_phase_quality_report(state)
    removed = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row_for(report)["review_id"]],
        decision="auto_remove",
        reason="test",
        backup_root=tmp_path,
    )["state"]
    assert sid not in manager.data["structures"]
    assert manager.data["branches"][bid]["structure_ids"] == [peer]
    assert not any(r["record_id"] == "r1" for r in removed["phase_records"])
    assert removed["phase_quality_archived_records"][0]["record_id"] == "r1"
    assert manager.add_structure(branch_id=bid, arrangement={"positions": [1]}) == sid
    from phase_agent.analysis.phase.phase_quality_disposition import filter_excluded_candidates

    fresh = {
        "structure_id": sid,
        "structure_sha256": "g1",
        "source_search_task_id": "new-task",
        "model_version": "m1",
    }
    assert filter_excluded_candidates(removed, [fresh]) == [fresh]
    same_geometry = {"structure_id": sid, "structure_sha256": "g1", "model_version": "m1"}
    assert filter_excluded_candidates(removed, [same_geometry]) == [same_geometry]
    assert not filter_excluded_candidates(removed, [{"record_id": "r1", "structure_sha256": "g1"}])


def test_unselected_valid_record_retains_shared_structure_registration(tmp_path):
    state, manager, _, _, bid, sid = fixture()
    other = deepcopy(state["phase_records"][1])
    other.update(record_id="old-model", model_version="m0", source_version="m0")
    state["phase_records"].append(other)
    report = get_phase_quality_report(state)
    removed = apply_quality_review(
        state,
        manager,
        report,
        review_ids=[row_for(report)["review_id"]],
        decision="auto_remove",
        reason="test",
        backup_root=tmp_path,
    )["state"]
    assert (
        sid in manager.data["structures"] and sid in manager.data["branches"][bid]["structure_ids"]
    )
    assert any(r["record_id"] == "old-model" for r in removed["phase_records"])


def test_semantic_policy_contract_and_config_only_turn():
    from phase_agent.decisions.agent.dialogue_contract import (
        dialogue_errors,
        is_dialogue,
        dialogue_output_contract,
    )

    action = {
        "kind": "phase_quality_policy",
        "cutoffs": {"h_cutoff": 60, "d_cutoff": 75},
        "source": "human",
        "reason": "用户指定",
    }
    assert is_dialogue(action) and not dialogue_errors(action, True)
    assert "phase_quality_policy_schema" in dialogue_output_contract(allow_controls=True)
    assert dialogue_errors({**action, "execute": True}, True)


def test_two_panel_preview_retains_formal_data_and_writes_all_formats(tmp_path):
    from phase_agent.analysis.phase.phase_quality_comparison import publish_quality_comparison

    state = sample((0, -0.075, 0, 0))
    before = deepcopy(state)
    report = diagnose_phase_diagram(state)
    assert row_for(report)["tier"] == 2
    result = publish_quality_comparison(state, report, directory=tmp_path, formats=("png", "pdf", "svg"))
    assert result["status"] == "completed" and len(result["plot_paths"]) == 3
    assert result["excluded_record_ids"] == ["r1"]
    assert state == before


def test_semantic_handler_updates_cutoff_with_one_model_call_and_no_science(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import write_json, read_json

    calls = []

    def model(payload):
        calls.append(payload)
        return {
            "kind": "phase_quality_policy",
            "cutoffs": {"d_cutoff": 75},
            "source": "human",
            "reason": "用户要求D阈值75",
        }

    handler = make_handler(tmp_path, model)
    state, manager, *_ = fixture()
    ledger = tmp_path / "ledgers" / "phase_data.json"
    manager.save(ledger)
    handler.workflow_kwargs["ledger_path"] = str(ledger)
    handler.workflow_kwargs["manager"] = manager
    write_json(handler.state_path, state)
    reply = handler(
        [{"role": "user", "content": "把同组分能量离群的门槛调到75 meV/O2"}],
        conversation_id="policy",
    )
    saved = read_json(handler.state_path, {})
    assert len(calls) == 1 and "已记录相图核查阈值" in reply
    assert saved["phase_quality_policy"]["overrides"]["outlier_gap_mev_per_O2"] == 75
    assert saved["phase_records"] == state["phase_records"]
    assert not saved.get("pending_execution_policies")


def test_simple_review_has_one_item_per_composition_phase():
    from phase_agent.analysis.phase.phase_quality_markdown import review_groups

    state = sample()
    add(state, "hidden", -0.2)
    report = diagnose_phase_diagram(state)
    groups = review_groups(report)
    assert len(groups) == len({(g["phase"], g["x"]) for g in groups})
    group = next(g for g in groups if g["x"] == 0.5 and g["phase"] == "O3")
    assert len(group["members"]) == 2
    assert group["H"] == pytest.approx(400)
    assert set(group["review_ids"]) == {
        r["review_id"] for r in report["structures"] if r["x"] == 0.5 and r["review_pending"]
    }
    from phase_agent.analysis.phase.phase_quality_reports import quality_summary

    summary = quality_summary(state)
    for target in summary["review_targets"]:
        parent = next(g for g in summary["review_groups"] if target["review_id"] in g["review_ids"])
        assert target["review_number"] == parent["number"]
