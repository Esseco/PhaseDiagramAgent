import json, zipfile
import pytest
from phase_agent.tools.relax_mc_batch import transition_imported_preview as module


def setup(tmp_path, monkeypatch):
    archive = tmp_path / "inputs.zip"
    batch = {"bundle_id": "b", "plan_checksum": "p", "model_version": "m", "config_version": "c"}
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("batch.json", json.dumps(batch))
    record = {
        **batch,
        "upload_archive": str(archive),
        "status": "partial_relax_imported",
        "branch_ids": ["x"],
        "threshold_transition": {
            "status": "relax_phase_imported",
            "phase_diagram_version": "v",
            "review_digest": "d",
        },
    }
    state = {
        "relax_mc_batches": {"b": record},
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m", "version": "v"}},
    }
    ledger = {"boundary": {}, "branches": {"x": {}, "outside": {}}, "structures": {}}

    class Manager:
        def __init__(self, *args):
            self.data = None

    monkeypatch.setattr(module, "PhaseDataManager", Manager)

    def scheduler(state, manager, manifest, root, index):
        assert set(manager.data["branches"]) == {"x"}
        return {"status": "scheduled", "actions": [{"branch_id": "x"}], "missing_branch_ids": []}

    monkeypatch.setattr(module, "next_mc", scheduler)
    return state, ledger


def test_preview_uses_imported_hull_and_original_scope(tmp_path, monkeypatch):
    state, ledger = setup(tmp_path, monkeypatch)
    result = module.preview(state, ledger, "b")
    assert (
        result["mc_action_count"] == 1
        and result["project_tasks_created"] is False
        and result["submitted"] is False
    )
    assert set(ledger["branches"]) == {"x", "outside"}


def test_preview_refuses_changed_hull(tmp_path, monkeypatch):
    state, ledger = setup(tmp_path, monkeypatch)
    state["phase_diagrams"]["mlip"]["version"] = "different"
    with pytest.raises(ValueError, match="hull_changed"):
        module.preview(state, ledger, "b")
