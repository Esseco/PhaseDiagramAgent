from copy import deepcopy
import json
from types import SimpleNamespace

from phase_agent.tools.workflows import replace_initial_pool as module


def frame(tmp_path):
    root = tmp_path / "candidates"
    root.mkdir()
    old = root / "old.vasp"
    old.write_text("old")
    unrelated = root / "notes.txt"
    unrelated.write_text("keep")
    manager = SimpleNamespace(
        data={"branches": {"old": {"P": "O1"}}, "structures": {"s": {"source_path": str(old)}}}
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(manager.data))
    manager.save = lambda path: path.write_text(json.dumps(manager.data))
    state = {"tasks": [], "generation_history": [{"task_key": "old", "new_branch_ids": ["old"]}]}
    context = {
        "manager": manager,
        "event_state": state,
        "effective_config": {"structure_directory": str(root), "ledger_path": ledger},
        "phase_references": {},
    }
    action = {
        "parameters": {
            "replace_initial_pool": True,
            "max_new_branches": 500,
            "regions": [{"phase": "O1"}],
        }
    }
    return context, action, old, unrelated, ledger


def test_replacement_refuses_calculated_pool(tmp_path):
    context, action, old, _, _ = frame(tmp_path)
    context["event_state"]["tasks"] = [{"task_id": "real"}]
    result = module.replace_initial_pool(action=action, context=context, generate=lambda **kw: None)
    assert result["status"] == "rejected"
    assert old.exists()


def test_replacement_failure_preserves_ledger_and_files(tmp_path, monkeypatch):
    context, action, old, _, ledger = frame(tmp_path)
    original = ledger.read_bytes()
    monkeypatch.setattr(module, "preview", lambda *a: {})
    result = module.replace_initial_pool(
        action=action, context=context, generate=lambda **kw: {"status": "rejected"}
    )
    assert result["status"] == "rejected"
    assert ledger.read_bytes() == original and old.exists()


def test_replacement_removes_only_owned_files_after_success(tmp_path, monkeypatch):
    context, action, old, unrelated, ledger = frame(tmp_path)
    monkeypatch.setattr(module, "preview", lambda *a: {})

    def generate(**kw):
        assert kw["context"]["effective_config"]["ledger_path"] is None
        assert not kw["context"]["manager"].data["branches"]
        kw["context"]["manager"].data = {"branches": {"new": {"P": "O1"}}, "structures": {}}
        # This fake manager's save callback must use its own new data.
        mgr = kw["context"]["manager"]
        mgr.save = lambda path: path.write_text(json.dumps(mgr.data))
        return {
            "status": "completed",
            "state": deepcopy(kw["context"]["event_state"]),
            "summary": {},
        }

    result = module.replace_initial_pool(action=action, context=context, generate=generate)
    assert result["status"] == "completed"
    assert not old.exists() and unrelated.read_text() == "keep"
    assert set(json.loads(ledger.read_text())["branches"]) == {"new"}
    assert result["summary"]["replaced_branch_count"] == 1
