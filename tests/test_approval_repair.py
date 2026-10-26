from copy import deepcopy
import hashlib
import json

import pytest

from phase_agent.tools.state.workflow_control import FIELD, commit_control, reconciliation_patch
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.graphs.business_checkpoint import restore_frame
from phase_agent.tools.state.execution_receipts import begin_execution, recovery_report
from phase_agent.tools.state.execution_reconciliation import (
    propose_reconciliation,
    apply_reconciliation,
)
from phase_agent.tools.workflows.generation_artifacts import artifact_path


def control_fixture(tmp_path):
    path = tmp_path / "workflow_state/state.json"
    old = {
        "confirmed_config_version": "c",
        "active_model_version": "m",
        "pending_execution_policies": {"old": {"kind": "old_quota"}},
        "tasks": [{"task_id": "keep"}],
    }
    write_json(path, old)
    patch = {
        "values": {"branch_count_policy": {"version": "region-adaptive-branch-v2"}},
        "pending_remove": ["old"],
        "pending_add": {"new": {"kind": "region"}},
    }
    current = commit_control(
        path,
        old,
        patch,
        evidence={"user": "repair old approvals"},
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return path, old, current


def test_stale_writer_cannot_restore_old_approval(tmp_path):
    path, old, current = control_fixture(tmp_path)
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="stale_workflow_control_revision"):
        write_json(path, old)
    assert path.read_bytes() == saved
    assert current["tasks"] == old["tasks"]


def test_old_process_overwrite_is_repaired_when_reading(tmp_path):
    path, old, current = control_fixture(tmp_path)
    path.write_text(json.dumps(old), encoding="utf-8")
    restored = read_json(path)
    assert restored[FIELD] == current[FIELD]
    assert set(restored["pending_execution_policies"]) == {"new"}
    assert restored["tasks"] == old["tasks"]


def test_consumed_new_approval_is_not_reintroduced(tmp_path):
    path, old, current = control_fixture(tmp_path)
    current["pending_execution_policies"].clear()
    write_json(path, current)
    assert read_json(path)["pending_execution_policies"] == {}


def test_successive_control_revisions_keep_tombstones_without_reviving_consumed_new_plan(tmp_path):
    path, old, current = control_fixture(tmp_path)
    current["pending_execution_policies"].clear()
    write_json(path, current)
    commit_control(
        path,
        current,
        {"values": {"workflow_stage": "branch_generation_review"}},
        evidence={"new_review": True},
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    path.write_text(json.dumps(old), encoding="utf-8")
    restored = read_json(path)
    assert restored["pending_execution_policies"] == {}
    assert restored["branch_count_policy"]["version"] == "region-adaptive-branch-v2"


def test_financial_reconciliation_refuses_to_guess_from_an_older_overwrite(tmp_path):
    path, old, current = control_fixture(tmp_path)
    desired = deepcopy(current)
    desired["reserved_relative_cost"] = 0
    saved = commit_control(
        path,
        current,
        {},
        evidence={"verified_settlement": "test"},
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        desired_state=desired,
        restore_allowed=False,
    )
    assert read_json(path)["reserved_relative_cost"] == 0
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="stale_business_state_after_result_or_financial_reconciliation"):
        read_json(path)


def test_cross_revision_frame_refused():
    with pytest.raises(ValueError, match="stale_lifecycle_checkpoint"):
        restore_frame({"loaded_state": {FIELD: "new"}}, {"loaded_state": {"pending": "old"}})


def test_cached_response_partitioned_after_control_change(tmp_path):
    from phase_agent.graphs.project.runner import run_search_workflow_graph

    path, old, current = control_fixture(tmp_path)
    calls = []

    def initialize():
        facts = read_json(path)
        calls.append(facts.get("marker", "first"))
        return {
            "_workflow_prepared": True,
            "snapshot": {"config_version": "c"},
            "loaded_state": facts,
        }

    stages = dict(
        initialize=initialize,
        collect=lambda f: f,
        analyze=lambda f: f,
        wait=lambda f: f,
        assess=lambda f: f,
        act=lambda f, **kw: f,
        finalize=lambda f: {"status": "done", "marker": f["loaded_state"].get("marker", "first")},
    )
    kw = dict(
        checkpoint_path=tmp_path / "lifecycle.sqlite",
        business_state_path=path,
        invocation_id="same-request",
        project_thread="project",
        **stages,
    )
    assert run_search_workflow_graph(**kw)["marker"] == "first"
    assert run_search_workflow_graph(**kw)["marker"] == "first"
    assert len(calls) == 1
    current["marker"] = "second"
    write_json(path, current)
    # New administrative revision, even if the request id is reused.
    updated = commit_control(
        path,
        current,
        {"values": {"workflow_stage": "branch_generation_review"}},
        evidence={"new": "review"},
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert updated[FIELD] != current[FIELD]
    assert run_search_workflow_graph(**kw)["marker"] == "second"
    assert calls == ["first", "second"]


def branch_fixture(tmp_path):
    root = tmp_path / "project"
    path = root / "workflow_state/state.json"
    state = {
        "workspace_layout": {"workspace_root": str(root)},
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "pending_execution_policies": {},
        "tasks": [],
    }
    action = {"tool": "generate_branches", "task_key": "key", "parameters": {}}
    identity = {
        "invocation_id": "old",
        "tool": "generate_branches",
        "config_version": "c",
        "action_hash": "sha256:" + "a" * 64,
    }
    assert begin_execution(path, identity, action=action)["allowed"]
    source = root / "structures/s.vasp"
    source.parent.mkdir(parents=True)
    source.write_text("test scientific source")
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ledger = {
        "branches": {"b": {"P": "O3", "H": h, "x": "1", "T": ["Fe"], "structure_ids": ["s"]}},
        "structures": {"s": {"branch_id": "b", "source_path": str(source)}},
    }
    write_json(path.parent / "ledgers/phase_data.json", ledger)
    inventory = artifact_path(state, "key", {})
    raw = {
        "status": "completed",
        "task_key": "key",
        "model_version": "m",
        "config_version": "c",
        "actual": {
            "new_branch_count": 1,
            "registered_branch_count": 1,
            "registered_structure_count": 1,
            "branches": [
                {
                    "branch_id": "b",
                    "is_new": True,
                    "phase": "O3",
                    "H": h,
                    "Na_fraction": "1",
                    "structure_count": 1,
                    "structures": [{"structure_id": "s", "path": str(source)}],
                }
            ],
        },
    }
    write_json(inventory, raw)
    state["pending_execution_policies"]["old"] = {
        "agent_proposal": {"raw_action": action},
        "record_id": "old",
    }
    state["action_records"] = [
        {
            "record_id": "old",
            "status": "awaiting_approval",
            "agent_proposal": {"raw_action": action},
        }
    ]
    write_json(path, state)
    return path, state, identity, source, inventory


ATTEST = {
    "evidence_note": "Verified bound inventory, ledger membership and all source files. Cancel downstream uncompleted scope.",
    "external_jobs": "not_applicable",
    "output_inventory": "registered_outputs",
    "external_cost": None,
}


def test_registered_branch_effects_reconcile_without_calculation_tasks(tmp_path):
    path, state, identity, source, inventory = branch_fixture(tmp_path)
    receipt = (path.parent / "execution_receipts.sqlite").read_bytes()
    proposed = propose_reconciliation(path, state, "old", "verified_registered_effects", ATTEST)
    checks = proposed["state"]["pending_execution_recoveries"][proposed["plan_id"]]["proposal"][
        "checks"
    ]
    assert checks["task_count"] == 0
    assert checks["registered_branch_outputs"][0]["registered_structure_count"] == 1
    result = apply_reconciliation(path, proposed["state"], proposed["plan_id"], "confirm_sensitive")
    assert recovery_report(path, result["state"])["status"] == "clear"
    assert not result["state"]["pending_execution_policies"]
    assert result["state"]["action_records"][0]["status"] == "reconciled_existing_effects"
    assert isinstance(result["state"]["invocations"]["old"], dict)
    assert result["state"]["dedup_gate"]["valid_structure_ids"] == ["s"]
    assert not begin_execution(path, identity)["allowed"]
    assert (path.parent / "execution_receipts.sqlite").read_bytes() == receipt


@pytest.mark.parametrize("change", ["source", "identity", "count", "ledger"])
def test_forged_or_missing_branch_evidence_cannot_clear_receipt(tmp_path, change):
    path, state, identity, source, inventory = branch_fixture(tmp_path)
    if change == "source":
        source.unlink()
    elif change == "ledger":
        ledger_path = path.parent / "ledgers/phase_data.json"
        doc = json.loads(ledger_path.read_text())
        doc["structures"].clear()
        write_json(ledger_path, doc)
    else:
        doc = json.loads(inventory.read_text())
        if change == "identity":
            doc["task_key"] = "other"
        else:
            doc["actual"]["registered_structure_count"] = 2
        write_json(inventory, doc)
    with pytest.raises(ValueError, match="branch"):
        propose_reconciliation(path, state, "old", "verified_registered_effects", ATTEST)
    assert recovery_report(path, state)["unsettled"]


def test_branch_outputs_cannot_be_declared_no_effect(tmp_path):
    path, state, *_ = branch_fixture(tmp_path)
    with pytest.raises(ValueError):
        propose_reconciliation(
            path,
            state,
            "old",
            "verified_no_effect",
            {**ATTEST, "output_inventory": "checked_no_outputs", "external_cost": 0},
        )


def test_evidence_changes_after_review_rejected(tmp_path):
    path, state, identity, source, inventory = branch_fixture(tmp_path)
    proposed = propose_reconciliation(path, state, "old", "verified_registered_effects", ATTEST)
    source.write_text("changed after review")
    with pytest.raises(ValueError, match="evidence_changed"):
        apply_reconciliation(path, proposed["state"], proposed["plan_id"], "confirm_sensitive")


def test_pending_region_review_blocks_downstream_even_with_old_generated_branches():
    from phase_agent.tools.policy.branch_conditions import branch_entry_errors

    state = {"branch_region_review": {"status": "awaiting_approval"}}
    errors = branch_entry_errors(
        {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_mc_inputs"}}, state
    )
    assert "region_branch_review_required_before_downstream_inputs" in errors


def test_region_approval_card_shows_regions_instead_of_zero_targets():
    from phase_agent.runtime.review_presentation import review_card

    card = review_card(
        "new",
        {
            "agent_proposal": {
                "raw_action": {
                    "tool": "generate_branches",
                    "parameters": {"sampling_mode": "region_adaptive", "regions": [{}, {}]},
                }
            }
        },
    )
    assert card["target_count"] is None and card["region_count"] == 2
    assert "重点区域 2" in card["scope"]
    assert "不准备Relax/MC输入" in card["approval_boundary"]


def test_new_region_result_is_recoverable_and_proposal_does_not_dump_cell_matrices(tmp_path):
    from types import SimpleNamespace
    from phase_agent.tools.workflows.region_review import write_proposal, write_result
    from phase_agent.tools.state.registered_branch_evidence import inspect_registered_branches

    path, state, identity, source, inventory = branch_fixture(tmp_path)
    ledger = json.loads((path.parent / "ledgers/phase_data.json").read_text())
    region = {
        "region_id": "r",
        "priority": 1,
        "strategy": "coverage",
        "phase": "O3",
        "min_det_H": 1,
        "max_det_H": 1,
        "na_min": 1,
        "na_max": 1,
        "reason": "test",
        "evidence_refs": [],
    }
    action = {
        "tool": "generate_branches",
        "task_key": "new-region",
        "parameters": {
            "sampling_mode": "region_adaptive",
            "regions": [region],
            "region_preview": {
                "regions": [
                    {"region_id": "r", "cells": [{"H": "large matrix repeated"}], "legal_cells": 1}
                ]
            },
        },
    }
    artifact = write_proposal({"raw_action": action}, state)["json"]
    doc = json.loads(__import__("pathlib").Path(artifact).read_text(encoding="utf-8"))
    assert "cells" not in doc["程序记录（可折叠）"]["parameters"]["region_preview"]["regions"][0]
    write_result(
        {"new_branch_ids": ["b"], "summary": {}},
        SimpleNamespace(data=ledger),
        [{"branch_id": "b", "structure_id": "s"}],
        state,
        action,
    )
    evidence = inspect_registered_branches(state, action, path)
    assert evidence["status"] == "verified"
    assert evidence["registered_structure_count"] == 1
