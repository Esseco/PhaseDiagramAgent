import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.policies.file_approval import proposal_hash
from phase_agent.policies.region_sampling import ledger_fingerprint
from phase_agent.tools.state.abandoned_region_generation import inspect_abandoned_region_generation
from phase_agent.tools.state.execution_receipts import begin_execution, record_execution_return, recovery_report
from phase_agent.tools.state.execution_reconciliation import propose_reconciliation, apply_reconciliation
from phase_agent.tools.workflows.region_review import path_for


def setup_case(tmp_path):
    root = tmp_path / "project"
    state_path = root / "workflow_state" / "state.json"
    state_path.parent.mkdir(parents=True)
    ledger = {"branches": {}, "structures": {}}
    ledger_path = state_path.parent / "ledgers" / "phase_data.json"
    ledger_path.parent.mkdir()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    key = "region-old"
    action = {"tool": "generate_branches", "task_key": key, "parameters": {
        "sampling_mode": "region_adaptive", "initial_states_per_branch": 1,
        "region_preview": {"ledger_fingerprint": ledger_fingerprint(SimpleNamespace(data=ledger))}}}
    identity = {"invocation_id": "old", "config_version": "c", "tool": "generate_branches", "action_hash": proposal_hash(action)}
    state = {"active_model_version": "m", "confirmed_config_version": "c",
             "confirmed_config": {"storage": {"workspace_root": str(root)}},
             "action_records": [{"record_id": "old", "status": "failed", "final_action": action,
                "human_feedback": {"decision": "approve"}, "execution_result": {"error": "MemoryError: "}}],
             "generation_history": [], "tasks": [], "phase_records": [],
             "budget_reservations": {key: {"status": "reserved", "stage": "generate_branches", "reserved_cost": 0}},
             "reserved_relative_cost": 0, "effective_decisions": {key: {"status": "reconciliation_required"}},
             "pending_execution_policies": {}, "invocations": {}}
    directory = root / "structures" / "candidate_structures" / ("RegionSampling-" + hashlib.sha256(key.encode()).hexdigest()[:16])
    folder = directory / "B-123456789abc"; folder.mkdir(parents=True)
    (folder / "full_na_structure.vasp").write_text("full", encoding="utf-8")
    (folder / "S-123456789abc.vasp").write_text("frame", encoding="utf-8")
    review = path_for(state, action); review.parent.mkdir(parents=True)
    review.write_text(json.dumps({"程序记录（可折叠）": {"status": "awaiting_approval", "task_key": key,
                       "model_version": "m", "config_version": "c", "actual": None}}, ensure_ascii=False), encoding="utf-8")
    begin_execution(state_path, identity, action=action)
    record_execution_return(state_path, identity, "failed", evidence={"error": "MemoryError: ", "task_ids": [], "artifacts": []})
    return state_path, state, identity, directory, ledger_path


def test_region_staging_is_reviewed_without_deleting_or_registering(tmp_path):
    path, state, identity, directory, _ = setup_case(tmp_path)
    before = {str(p): p.read_bytes() for p in directory.rglob("*.vasp")}
    attestation = {"evidence_note": "用户要求弃用已核验的未登记暂存，文件保留追溯", "external_jobs": "not_applicable",
                   "output_inventory": "abandoned_local_staging", "external_cost": 0}
    result = propose_reconciliation(path, state, "old", "verified_abandoned_staging", attestation)
    plan = result["state"]["pending_execution_recoveries"][result["plan_id"]]["proposal"]
    assert plan["checks"]["abandoned_staging"]["file_count"] == 2
    with pytest.raises(ValueError, match="confirm_sensitive"):
        apply_reconciliation(path, result["state"], result["plan_id"], "approve")
    accepted = apply_reconciliation(path, result["state"], result["plan_id"], "confirm_sensitive")
    assert accepted["status"] == "reconciled_abandoned_staging"
    assert recovery_report(path, accepted["state"])["status"] == "clear"
    assert not accepted["state"]["generation_history"]
    assert accepted["state"]["budget_reservations"]["region-old"]["status"] == "settled"
    assert {str(p): p.read_bytes() for p in directory.rglob("*.vasp")} == before
    assert not begin_execution(path, identity)["allowed"]


def test_region_staging_changes_are_rejected_before_approval(tmp_path):
    path, state, identity, directory, _ = setup_case(tmp_path)
    att = {"evidence_note": "保留并弃用暂存", "external_jobs": "not_applicable", "output_inventory": "abandoned_local_staging", "external_cost": 0}
    proposed = propose_reconciliation(path, state, "old", "verified_abandoned_staging", att)
    next(directory.rglob("S-*.vasp")).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="recovery_evidence_changed"):
        apply_reconciliation(path, proposed["state"], proposed["plan_id"], "confirm_sensitive")


@pytest.mark.parametrize("change", ["ledger", "registered", "extra_file", "action", "task"])
def test_region_staging_rejects_unproven_scope(tmp_path, change):
    path, state, identity, directory, ledger = setup_case(tmp_path)
    if change == "ledger":
        ledger.write_text(json.dumps({"branches": {}, "structures": {"other": {}}}), encoding="utf-8")
    elif change == "registered":
        state["generation_history"].append({"task_key": "region-old"})
    elif change == "extra_file":
        (next(directory.iterdir()) / "unknown.txt").write_text("x")
    elif change == "action":
        state["action_records"][0]["human_feedback"]["decision"] = "reject"
    elif change == "task":
        state["tasks"].append({"branch_id": "B-123456789abc"})
    with pytest.raises(ValueError):
        inspect_abandoned_region_generation(path, state, identity)


def test_region_staging_review_card_describes_scope(tmp_path):
    from phase_agent.runtime.review_requests import additional_reviews
    path, state, _, _, _ = setup_case(tmp_path)
    att = {"evidence_note": "保留并弃用暂存", "external_jobs": "not_applicable", "output_inventory": "abandoned_local_staging", "external_cost": 0}
    result = propose_reconciliation(path, state, "old", "verified_abandoned_staging", att)
    row = next(r for r in additional_reviews(result["state"]) if r["plan_id"] == result["plan_id"])
    assert "弃用 1 个未登记结构" in row["review_card"]["scope"]
    assert row["review_card"]["sensitive"] is True


def test_region_staging_cannot_settle_nonzero_cost(tmp_path):
    path, state, _, _, _ = setup_case(tmp_path)
    state["budget_reservations"]["region-old"]["reserved_cost"] = 1
    state["reserved_relative_cost"] = 1
    att = {"evidence_note": "保留并弃用暂存", "external_jobs": "not_applicable", "output_inventory": "abandoned_local_staging", "external_cost": 0}
    with pytest.raises(ValueError, match="region_staging_requires_verified"):
        propose_reconciliation(path, state, "old", "verified_abandoned_staging", att)
