import json, threading
from pathlib import Path
from types import SimpleNamespace
import pytest
from phase_agent.runtime import relax_mc_phase_import as module


def setup(tmp_path, monkeypatch):
    root = tmp_path / "workflow_state"
    ledger = root / "ledgers/phase_data.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"boundary": {}, "branches": {}, "structures": {}}), encoding="utf8"
    )
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "tasks": [{"task_id": "t", "status": "pending", "relax_mc_bundle_id": "b"}],
        "relax_mc_batches": {
            "b": {
                "status": "prepared",
                "threshold_transition": {
                    "status": "verified_awaiting_continuation",
                    "task_count": 1,
                    "review_digest": "d",
                    "raw_archive_path": "raw",
                    "raw_archive_sha256": "x",
                    "scheduler_receipts_path": "jobs",
                    "scheduler_receipts_sha256": "y",
                },
            }
        },
    }
    path = root / "state.json"
    path.write_text(json.dumps(state), encoding="utf8")
    ctl = SimpleNamespace(
        state_path=path,
        chat_handler=SimpleNamespace(
            lock=threading.RLock(),
            workflow_kwargs={"ledger_path": str(ledger), "phase_references": {"O3": "ref"}},
        ),
        _codex_context_cache=None,
    )
    review = {
        "review_digest": "d",
        "raw_archive_sha256": "x",
        "model_version": "m",
        "task_count": 1,
        "effective_failure_count": 0,
        "identity_check": {"mismatches": []},
    }
    monkeypatch.setattr(module, "verify_transition", lambda *a: review)
    monkeypatch.setattr(
        module, "materialize_verified_relax", lambda *a: {"results": [{"task_id": "t"}]}
    )

    class Manager:
        def __init__(self, *a):
            self.data = None

    monkeypatch.setattr(module, "PhaseDataManager", Manager)

    def reconcile(s, r):
        updated = json.loads(json.dumps(s))
        updated["tasks"][0]["status"] = "completed"
        return {"state": updated, "reconciled": [{"status": "settled"}]}

    monkeypatch.setattr(module, "reconcile_task_results", reconcile)

    def feedback(s, r, **k):
        s["phase_diagrams"] = {
            "mlip": {"status": "completed", "model_version": "m", "version": "v", "entries": [{}]}
        }
        return {"state": s, "recorded": ["t"], "rejected": []}

    monkeypatch.setattr(module, "apply_scientific_feedback", feedback)
    return ctl, path, ledger


def test_atomic_import_and_replay(tmp_path, monkeypatch):
    ctl, path, ledger = setup(tmp_path, monkeypatch)
    out = module.import_relax_phase(ctl, "b")
    state = json.loads(path.read_text(encoding="utf8"))
    assert out["status"] == "relax_phase_imported" and out["submitted"] is False
    assert state["tasks"][0]["status"] == "completed"
    assert state["relax_mc_batches"]["b"]["status"] == "partial_relax_imported"
    assert module.import_relax_phase(ctl, "b")["status"] == "already_imported"


def test_feedback_rejection_never_commits(tmp_path, monkeypatch):
    ctl, path, ledger = setup(tmp_path, monkeypatch)
    before = path.read_bytes()
    oldledger = ledger.read_bytes()
    monkeypatch.setattr(
        module,
        "apply_scientific_feedback",
        lambda s, r, **k: {"state": s, "recorded": [], "rejected": [{"task_id": "t"}]},
    )
    with pytest.raises(ValueError, match="scientific_feedback_incomplete"):
        module.import_relax_phase(ctl, "b")
    assert path.read_bytes() == before and ledger.read_bytes() == oldledger
