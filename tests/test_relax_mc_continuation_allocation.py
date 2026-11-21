import json, threading, zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from phase_agent.runtime import relax_mc_continuation_allocation as module
from phase_agent.tools.remote.transfer_bundle import checksum


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "workflow_state"
    led = root / "ledgers/phase.json"
    led.parent.mkdir(parents=True)
    ledger = {"boundary": {}, "branches": {"x": {}}, "structures": {}}
    led.write_text(json.dumps(ledger), encoding="utf8")
    frozen = {
        "bundle_id": "b",
        "plan_checksum": "p",
        "model_version": "m",
        "config_version": "c",
        "workload": {"mc_task_upper": [1]},
    }
    archive = tmp_path / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("batch.json", json.dumps(frozen))
    original = {
        "tool": "prepare_local_batch_files",
        "target_ids": ["x"],
        "parameters": {
            "mode": "relax_mc_inputs",
            "relax_mc_preview": {
                "bundle_id": "b",
                "checksum": "p",
                "workload": {"mc_task_upper": [1]},
            },
        },
    }
    rec = {
        **frozen,
        "status": "partial_relax_imported",
        "branch_ids": ["x"],
        "upload_archive": str(archive),
        "threshold_transition": {
            "status": "relax_phase_imported",
            "review_digest": "d",
            "phase_diagram_version": "v",
            "raw_archive_path": "raw",
            "raw_archive_sha256": "r",
            "scheduler_receipts_path": "jobs",
            "scheduler_receipts_sha256": "j",
        },
    }
    s = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "relax_mc_batches": {"b": rec},
        "phase_diagrams": {"mlip": {"version": "v"}},
        "tasks": [],
        "action_records": [
            {
                "record_id": "approved",
                "human_feedback": {"decision": "approve"},
                "final_action": original,
            }
        ],
    }
    path = root / "state.json"
    path.write_text(json.dumps(s), encoding="utf8")
    action = {"branch_id": "x", "structure_id": "s", "structure_path": str(tmp_path / "s.vasp")}
    Path(action["structure_path"]).write_text("structure")
    preview = {
        "bundle_id": "b",
        "review_digest": "d",
        "hull_version": "v",
        "submitted": False,
        "project_tasks_created": False,
        "actions": [action],
        "missing_branch_ids": [],
    }
    pp = tmp_path / "preview.json"
    pp.write_text(json.dumps(preview), encoding="utf8")
    review = {
        "review_digest": "d",
        "identity_check": {"mismatches": []},
        "rows": [
            {
                "structure_id": "s",
                "branch_id": "x",
                "final_sha256": checksum(action["structure_path"]),
            }
        ],
    }
    monkeypatch.setattr(module, "verify_transition", lambda *args: review)

    class Manager:
        def __init__(self, *args):
            self.data = None

    monkeypatch.setattr(module, "PhaseDataManager", Manager)

    def next_mc(state, manager, manifest, directory, index):
        value = json.loads(json.dumps(state))
        value["tasks"].append(
            {
                "task_id": "mc",
                "relax_mc_bundle_id": "b",
                "stage": "deep_search",
                "status": "pending",
            }
        )
        return {
            "status": "scheduled",
            "actions": [action],
            "missing_branch_ids": [],
            "state": value,
        }

    monkeypatch.setattr(module, "next_mc", next_mc)
    ctl = SimpleNamespace(
        state_path=path,
        chat_handler=SimpleNamespace(
            lock=threading.RLock(), workflow_kwargs={"ledger_path": str(led)}
        ),
        _codex_context_cache=None,
    )
    return ctl, path, pp


def test_allocation_is_bound_to_approved_scope_and_idempotent(tmp_path, monkeypatch):
    ctl, path, preview = fixture(tmp_path, monkeypatch)
    out = module.allocate(
        ctl, bundle_id="b", preview_path=preview, preview_sha256=checksum(preview)
    )
    assert out["task_count"] == 1 and out["submitted"] is False
    saved = json.loads(path.read_text(encoding="utf8"))
    assert saved["tasks"][0]["stage"] == "deep_search"
    assert saved["relax_mc_batches"]["b"]["status"] == "partial_relax_imported"
    assert (
        module.allocate(ctl, bundle_id="b", preview_path=preview, preview_sha256=checksum(preview))[
            "status"
        ]
        == "already_allocated"
    )


def test_unapproved_scope_and_changed_preview_never_commit(tmp_path, monkeypatch):
    ctl, path, preview = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="preview_changed"):
        module.allocate(ctl, bundle_id="b", preview_path=preview, preview_sha256="bad")
    state = json.loads(path.read_text(encoding="utf8"))
    state["action_records"][0]["human_feedback"]["decision"] = "reject"
    path.write_text(json.dumps(state), encoding="utf8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="approval_missing"):
        module.allocate(ctl, bundle_id="b", preview_path=preview, preview_sha256=checksum(preview))
    assert path.read_bytes() == before
