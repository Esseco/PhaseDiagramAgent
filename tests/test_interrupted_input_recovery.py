from copy import deepcopy

import pytest

from phase_agent.tools.policy.file_approval import proposal_hash
from phase_agent.tools.state.execution_receipts import begin_execution, recovery_report
from phase_agent.tools.state.execution_reconciliation import (
    apply_reconciliation,
    propose_reconciliation,
)
from tests.test_recovery_closure import ATTESTATION


def input_case(tmp_path):
    action = {
        "tool": "prepare_local_batch_files",
        "task_key": "inputs:one",
        "parameters": {"mode": "relax_mc_inputs", "relax_mc_preview": {"checksum": "same"}},
    }
    identity = {
        "tool": action["tool"],
        "config_version": "c1",
        "invocation_id": "old-input",
        "action_hash": proposal_hash(action),
    }
    pending = {
        "record_id": "old-input",
        "revision": 0,
        "feedback_history": [],
        "project_identity": {"config_version": "c1", "model_version": "m1"},
        "agent_proposal": {"raw_action": action},
    }
    state = {
        "confirmed_config_version": "c1",
        "active_model_version": "m1",
        "tasks": [{"task_id": "unrelated", "status": "completed"}],
        "pending_execution_policies": {"old-input": pending},
        "action_records": [
            {
                "record_id": "old-input",
                "status": "awaiting_approval",
                "agent_proposal": deepcopy(pending["agent_proposal"]),
            }
        ],
    }
    path = tmp_path / "state.json"
    assert begin_execution(path, identity, action=action)["allowed"]
    return path, state, identity


def test_reviewed_no_effect_reoffers_exact_input_plan_with_new_identity(tmp_path):
    path, state, identity = input_case(tmp_path)
    before = deepcopy(state)
    plan = propose_reconciliation(
        path, state, identity["invocation_id"], "verified_no_effect", ATTESTATION
    )
    result = apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive")
    updated = result["state"]
    assert state == before
    assert recovery_report(path, updated)["status"] == "clear"
    assert not begin_execution(path, identity)["allowed"]
    pending = updated["pending_execution_policies"]
    assert len(pending) == 1 and "old-input" not in pending
    row = next(iter(pending.values()))
    assert (
        row["agent_proposal"] == before["pending_execution_policies"]["old-input"]["agent_proposal"]
    )
    assert row["recovery_source_record_id"] == "old-input"
    assert row["feedback_history"] == [] and row.get("human_feedback") is None
    assert updated["tasks"] == before["tasks"]
    assert result["restored_plan_id"] in pending


def test_reoffered_plan_survives_control_revision_restore(tmp_path):
    from phase_agent.tools.state.workflow_control import (
        reconciliation_patch,
        commit_control,
        restore_control,
    )
    from phase_agent.tools.step_runner.file_protocol import write_json
    import hashlib

    path, state, identity = input_case(tmp_path)
    write_json(path, state)
    plan = propose_reconciliation(
        path, state, identity["invocation_id"], "verified_no_effect", ATTESTATION
    )
    updated = apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive")[
        "state"
    ]
    committed = commit_control(
        path,
        state,
        reconciliation_patch(state, updated),
        evidence={"review": "explicit"},
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        desired_state=updated,
    )
    restored = restore_control(path, state)
    assert len(restored["pending_execution_policies"]) == 1
    assert restored["pending_execution_policies"] == committed["pending_execution_policies"]


@pytest.mark.parametrize("change", ["config", "model", "hash"])
def test_stale_frozen_plan_is_not_reoffered(tmp_path, change):
    path, state, identity = input_case(tmp_path)
    row = state["pending_execution_policies"]["old-input"]
    if change == "hash":
        row["agent_proposal"]["raw_action"]["parameters"]["mode"] = "other"
    else:
        row["project_identity"][change + "_version"] = "stale"
    plan = propose_reconciliation(path, state, "old-input", "verified_no_effect", ATTESTATION)
    updated = apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive")[
        "state"
    ]
    assert not updated["pending_execution_policies"]


def test_durable_progress_distinguishes_prewrite_and_partial_output(tmp_path):
    from phase_agent.tools.state.execution_progress import execution_scope, preparation_progress
    from phase_agent.tools.state.execution_recovery_evidence import inspect_execution

    path, state, identity = input_case(tmp_path)
    output = tmp_path / "batch"
    with execution_scope(path, identity):
        preparation_progress("started", {"protocol": 1, "mode": "relax_mc_inputs"})
        preparation_progress("plan_verified", {"directory": str(output), "bundle_id": "batch-id"})
    issue = recovery_report(path, state)["unsettled"][0]
    checks = inspect_execution(state, issue, state_path=path)
    assert checks["preparation_progress"]["boundary"] == "before_file_writes"
    with execution_scope(path, identity):
        preparation_progress("effects_started", {"directory": str(output)})
    output.mkdir()
    (output / "partial.txt").write_text("partial")
    checks = inspect_execution(state, issue, state_path=path)
    assert checks["preparation_progress"]["boundary"] == "effects_possible"
    assert any(item["path"] == str(output) for item in checks["artifact_checks"])
    with pytest.raises(ValueError, match="产出|文件|副作用"):
        propose_reconciliation(path, state, "old-input", "verified_no_effect", ATTESTATION)


@pytest.mark.parametrize("boundary", ["before_write", "after_mkdir"])
def test_real_builder_interruption_keeps_original_state_and_records_boundary(
    tmp_path, monkeypatch, boundary
):
    from tests.test_relax_mc_batch import fixture
    from phase_agent.tools.relax_mc_batch.plan import attach_preview
    from phase_agent.tools.relax_mc_batch.prepare import prepare_batch
    from phase_agent.tools.state.execution_progress import execution_scope
    from phase_agent.tools.state.execution_recovery_evidence import inspect_execution

    _, state, config, context, action = fixture(tmp_path, count=2)
    action = attach_preview(action, state, context, config)
    identity = {
        "tool": action["tool"],
        "config_version": "v1",
        "invocation_id": "interrupted-builder",
        "action_hash": proposal_hash(action),
    }
    path = tmp_path / "state.json"
    begin_execution(path, identity, action=action)
    before = deepcopy(state)

    def fail(*args, **kwargs):
        raise OSError("injected interruption")

    if boundary == "before_write":
        monkeypatch.setattr("phase_agent.graphs.cancellation.check_cancelled", fail)
    else:
        monkeypatch.setattr("phase_agent.tools.relax_mc_batch.prepare.shutil.copy2", fail)
    with pytest.raises(OSError, match="injected"):
        with execution_scope(path, identity):
            prepare_batch(action=action, context=context)
    assert state == before
    checks = inspect_execution(state, recovery_report(path, state)["unsettled"][0], state_path=path)
    assert checks["preparation_progress"]["boundary"] == (
        "before_file_writes" if boundary == "before_write" else "effects_possible"
    )
    assert not begin_execution(path, identity)["allowed"]


def test_dispatch_installs_and_clears_durable_execution_scope(tmp_path):
    from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
    from phase_agent.tools.state.execution_progress import preparation_progress
    from phase_agent.tools.state.execution_receipts import execution_events

    action = {"tool": "prepare_local_batch_files", "task_key": "dry-input", "parameters": {}}

    def handler(**kwargs):
        preparation_progress("started", {"protocol": 1, "mode": "test"})
        return {"status": "prepared", "submitted": False}

    path = tmp_path / "state.json"
    result = execute_tool_action(
        action,
        registry={action["tool"]: {"handler": handler}},
        context={"state_path": str(path), "invocation_id": "new-input", "config_version": "c1"},
    )
    assert result["status"] == "completed"
    journal = execution_events(path, "new-input")
    assert any(row["event"] == "input_preparation:started" for row in journal)
    preparation_progress("complete", {})
    assert execution_events(path, "new-input") == journal
