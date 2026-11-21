import hashlib
import json
from copy import deepcopy

import pytest

from phase_agent.tools.remote.integrity import payload_checksum
from phase_agent.tools.state.abandoned_refresh import inspect_abandoned_refresh
from phase_agent.tools.state.execution_reconciliation import _validate_checks
from phase_agent.tools.workflows.model_refresh_state import fingerprint


def fixture(tmp_path, monkeypatch):
    state_path = tmp_path / "workflow_state" / "state.json"
    directory = tmp_path / "submissions" / "epoch0" / "batch"
    directory.mkdir(parents=True)
    task_dir = directory / "00000-t"
    task_dir.mkdir()
    source = b"valid geometry"
    (task_dir / "initial.vasp").write_bytes(source)
    sha = hashlib.sha256(source).hexdigest()
    plan = {"refresh_scope": "frozen_baseline_hull_vertices", "new_model_version": "new",
            "candidates": [{"structure_id": "s", "structure_sha256": sha, "operation": "relax"}]}
    plan["checksum"] = fingerprint(plan)
    invocation = "refresh-plan:" + plan["checksum"] + ":0"
    action = {"tool": "prepare_local_batch_files", "task_key": invocation, "config_version": "C",
              "parameters": {"mode": "model_refresh_inputs", "preview_checksum": plan["checksum"], "refresh_preview": plan}}
    task = {"task_id": "t", "task_key": "model-refresh:" + plan["checksum"] + ":0:" + sha + ":relax",
            "structure_id": "s", "input_structure_sha256": sha, "model_refresh_id": plan["checksum"],
            "config_version": "C", "model_version": "new", "parent_decision_id": invocation, "status": "pending"}
    task["task_checksum"] = payload_checksum(task)
    task_path = task_dir / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf8")
    ref = {k: task[k] for k in ("task_id", "task_key", "config_version", "model_version", "parent_decision_id")}
    ref.update(input_path=str(task_path), result_path=str(tmp_path / "not-computed.json"))
    manifest = [{"task_id": "t", "task_key": task["task_key"], "input_path": "00000-t/task.json",
                 "config_version": "C", "model_version": "new", "task_checksum": task["task_checksum"]}]
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    snapshot = {"manifest_checksum": payload_checksum(manifest), "task_ids": ["t"], "config_version": "C", "model_version": "new"}
    (directory / "batch.snapshot.json").write_text(json.dumps(snapshot), encoding="utf8")
    (directory / "GPU.sh").write_text("script", encoding="utf8")
    ident = {"invocation_id": invocation, "tool": "prepare_local_batch_files"}
    evidence = {"result_status": "prepared", "returned_task_refs": [ref],
                "artifacts": [{"artifact_path": ref["input_path"]}, {"artifact_path": ref["result_path"]}]}
    events = [{"event": "claimed", "payload": {"action": action, "identity": ident}},
              {"event": "tool_returned", "payload": {"identity": ident, "outcome": "completed", "evidence": evidence}}]
    monkeypatch.setattr("phase_agent.tools.state.abandoned_refresh.execution_events", lambda *_: events)
    state = {"confirmed_config_version": "C", "active_model_version": "new", "tasks": [], "slurm_batches": [],
             "pending_execution_policies": {"P": {"agent_proposal": {"raw_action": deepcopy(action)}}}}
    issue = {"identity": {"invocation_id": invocation, "tool": "prepare_local_batch_files"}, "phase": "returned", "tool_outcome": "completed"}
    return state_path, state, issue, task_dir, events


def test_verified_staging_can_only_be_preserved_not_reused(tmp_path, monkeypatch):
    path, state, issue, task_dir, _ = fixture(tmp_path, monkeypatch)
    proof = inspect_abandoned_refresh(path, state, issue)
    assert proof["task_ids"] == ["t"] and proof["submitted"] is False
    checks = {"abandoned_staging": proof, "unregistered_returned_task_ids": ["t"], "task_count": 0, "budget_reservations": []}
    attestation = {"evidence_note": "remote checked", "external_jobs": "none_found", "output_inventory": "abandoned_local_staging", "external_cost": 0}
    _validate_checks(checks, "verified_abandoned_staging", attestation)
    assert (task_dir / "task.json").is_file()
    for field, value in [("external_jobs", "registered_only"), ("external_cost", 1), ("output_inventory", "registered_outputs")]:
        with pytest.raises(ValueError):
            _validate_checks(checks, "verified_abandoned_staging", {**attestation, field: value})
    with pytest.raises(ValueError):
        _validate_checks(checks, "verified_registered_effects", attestation)


@pytest.mark.parametrize("tamper", ["geometry", "task", "manifest", "config", "task_registered", "result"])
def test_staging_rejects_changed_or_executed_evidence(tmp_path, monkeypatch, tamper):
    path, state, issue, task_dir, events = fixture(tmp_path, monkeypatch)
    if tamper == "geometry": (task_dir / "initial.vasp").write_bytes(b"changed")
    if tamper == "task": (task_dir / "task.json").write_text("{}", encoding="utf8")
    if tamper == "manifest": (task_dir.parent / "manifest.json").write_text("[]", encoding="utf8")
    if tamper == "config": state["confirmed_config_version"] = "changed"
    if tamper == "task_registered": state["tasks"] = [{"task_id": "t"}]
    if tamper == "result": (tmp_path / "not-computed.json").write_text("{}", encoding="utf8")
    with pytest.raises((ValueError, KeyError)):
        inspect_abandoned_refresh(path, state, issue)


def test_review_card_handles_abandoned_refresh_task_count():
    from phase_agent.runtime.review_requests import additional_reviews
    state = {"pending_execution_recoveries": {"review:recovery:1": {"proposal": {
        "identity": {"invocation_id": "old"}, "resolution": "verified_abandoned_staging",
        "checks": {"abandoned_staging": {"kind": "abandoned_unsubmitted_refresh_staging", "task_count": 15}},
        "attestation": {"evidence_note": "checked"}}, "revision": 0}}}
    rows = additional_reviews(state)
    assert len(rows) == 1 and "15" in rows[0]["review_card"]["scope"]
