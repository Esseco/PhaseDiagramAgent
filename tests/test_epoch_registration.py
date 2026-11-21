import copy
import hashlib
from types import SimpleNamespace
import threading

import pytest

from phase_agent.runtime.epoch_registration import reconcile, reconcile_state
from phase_agent.tools.workflows.model_refresh_state import fingerprint


def fixture(tmp_path):
    archive = tmp_path / "inputs.zip"
    archive.write_bytes(b"frozen")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    plan = {"new_model_version": "new", "candidates": ["one", "two"]}
    plan["checksum"] = fingerprint(plan)
    checksum = plan["checksum"]
    handoff = {
        "batch_ids": ["b1"],
        "upload_archive": str(archive),
        "import_directory": str(tmp_path / "epoch1"),
        "return_directory": str(tmp_path / "epoch1" / "return"),
    }
    state = {
        "active_model_version": "new",
        "upload_layout": {"model_rounds": {"old": 2, "new": 3}, "scientific_epochs": {"new": 1}},
        "extrapolation_recovery_cycle": {
            "replacement_model_version": "new",
            "training_parent_model_version": "old",
            "status": "awaiting_new_model_refresh",
            "policy_version": "epoch_refresh_v1",
            "round_evidence_digest": "digest",
        },
        "model_refresh": {
            "new_model_version": "new",
            "status": "waiting_results",
            "handoff_status": "complete",
            "plan": plan,
            "approved_scope": {"plan_checksum": checksum, "model_version": "new"},
            "input_authorization": {"status": "approved", "plan": copy.deepcopy(plan)},
            "task_ids": ["t1", "t2"],
        },
        "tasks": [
            {
                "task_id": t,
                "batch_id": "b1",
                "status": "pending",
                "model_version": "new",
                "model_refresh_id": checksum,
                "input_path": f"epoch1/{t}",
            }
            for t in ("t1", "t2")
        ],
        "slurm_batches": [
            {
                "batch_id": "b1",
                "status": "prepared",
                "job_id": None,
                "model_version": "new",
                "handoff": handoff,
            }
        ],
    }
    args = dict(
        model_version="new",
        old_epoch=1,
        new_epoch=2,
        round_digest="digest",
        plan_checksum=checksum,
        archive_sha256=digest,
    )
    return state, args


def test_registration_preserves_frozen_artifacts_and_is_idempotent(tmp_path):
    state, args = fixture(tmp_path)
    original = copy.deepcopy(state)
    updated, receipt = reconcile_state(state, **args)
    assert receipt["status"] == "registered"
    assert updated["upload_layout"]["scientific_epochs"]["new"] == 2
    assert updated["tasks"] == original["tasks"]
    assert updated["slurm_batches"] == original["slurm_batches"]
    assert updated["model_refresh"]["plan"] == original["model_refresh"]["plan"]
    assert receipt["import_directory"].endswith("epoch1")
    _, again = reconcile_state(state, **args)
    assert again["status"] == "already_registered"


@pytest.mark.parametrize(
    "case",
    [
        "model",
        "digest",
        "plan",
        "archive",
        "submitted",
        "task_result",
        "result_file",
        "foreign_task",
        "conflict",
        "parent",
    ],
)
def test_registration_rejects_changed_evidence_without_mutation(tmp_path, case):
    state, args = fixture(tmp_path)
    if case == "model":
        args["model_version"] = "other"
    elif case == "digest":
        args["round_digest"] = "other"
    elif case == "plan":
        state["model_refresh"]["plan"]["candidates"].append("tampered")
    elif case == "archive":
        args["archive_sha256"] = "wrong"
    elif case == "submitted":
        state["slurm_batches"][0]["job_id"] = "123"
    elif case == "task_result":
        state["tasks"][0]["status"] = "completed"
    elif case == "result_file":
        result = tmp_path / "existing-result.json"
        result.write_text("{}", encoding="utf8")
        state["tasks"][0]["result_path"] = str(result)
    elif case == "foreign_task":
        state["tasks"].append({"task_id": "foreign", "batch_id": "b1"})
    elif case == "conflict":
        state["upload_layout"]["scientific_epochs"]["other"] = 2
    else:
        state["extrapolation_recovery_cycle"]["training_parent_model_version"] = "other"
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match="epoch_registration"):
        reconcile_state(state, **args)
    assert state == original


def test_control_uses_guarded_commit_and_detects_concurrent_change(tmp_path, monkeypatch):
    import json
    from phase_agent.persistence import result_recovery_commit

    state, args = fixture(tmp_path)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf8")
    control = SimpleNamespace(
        state_path=path,
        chat_handler=SimpleNamespace(lock=threading.RLock()),
        _codex_context_cache="old",
    )
    original = path.read_bytes()

    def reject(*args, **kwargs):
        raise ValueError("version_conflict")

    monkeypatch.setattr(result_recovery_commit, "commit_result_recovery", reject)
    with pytest.raises(ValueError, match="version_conflict"):
        reconcile(control, **args)
    assert path.read_bytes() == original


def test_control_commits_once_and_reuses_receipt(tmp_path):
    import json

    state, args = fixture(tmp_path)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf8")
    control = SimpleNamespace(
        state_path=path,
        chat_handler=SimpleNamespace(lock=threading.RLock()),
        _codex_context_cache="old",
    )
    assert reconcile(control, **args)["status"] == "registered"
    stored = json.loads(path.read_text(encoding="utf8"))
    assert stored["upload_layout"]["scientific_epochs"]["new"] == 2
    assert stored["tasks"] == state["tasks"]
    before = path.read_bytes()
    assert reconcile(control, **args)["status"] == "already_registered"
    assert path.read_bytes() == before


def test_mcp_entry_dispatches_and_rejects_extra_arguments():
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    class Control:
        def reconcile_recovery_refresh_epoch(self, **args):
            return {"status": "registered", **args}

    server = CodexMcpServer(SimpleNamespace(control=Control()))
    args = dict(
        model_version="new",
        old_epoch=1,
        new_epoch=2,
        round_digest="digest",
        plan_checksum="plan",
        archive_sha256="archive",
    )
    assert (
        server._call_tool("phase_reconcile_recovery_refresh_epoch", args)["status"] == "registered"
    )
    with pytest.raises(ValueError, match="invalid_epoch_registration_arguments"):
        server._call_tool("phase_reconcile_recovery_refresh_epoch", {**args, "extra": 1})
