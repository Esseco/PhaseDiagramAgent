"""Status and diagnostics must not restore archives or alter file attributes."""

from copy import deepcopy
import json

from phase_agent.runtime.status_presentation import format_progress
from phase_agent.tools.workflows.readonly_views import is_read_only
from phase_agent.tools.remote import submission_storage
from tests.test_remote_training_recovery import returned_job


def test_status_does_not_restore_storage_but_recovery_still_does(tmp_path, monkeypatch):
    from phase_agent.tools.local.recover_remote_training import recover_remote_training

    job, results = returned_job(tmp_path)
    packed = submission_storage.pack_storage_directory(results, "results")
    for path in list(results.iterdir()):
        if path.is_file() and str(path) != packed["archive"]:
            path.unlink()
    state = {"active_model_version": "base", "remote_finetune_jobs": {"j": job}}
    original = deepcopy(state)
    before = {str(p): p.read_bytes() for p in results.iterdir() if p.is_file()}
    hidden = []
    monkeypatch.setattr(submission_storage, "_hide", lambda path: hidden.append(path))
    format_progress(state)
    assert state == original
    assert hidden == []
    assert before == {str(p): p.read_bytes() for p in results.iterdir() if p.is_file()}
    recovered, reports = recover_remote_training(state)
    assert reports[0]["status"] == "validation_required"
    assert (results / "training.finished.json").is_file()
    assert hidden
    assert recovered["remote_finetune_jobs"]["j"]["status"] == "results_received"


def test_project_diagnostics_keeps_nested_presentation_in_readonly_context(tmp_path, monkeypatch):
    from phase_agent.runtime.project_diagnostics import inspect_project
    from phase_agent.runtime import status_presentation

    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    before = path.read_bytes()
    def presentation(state):
        assert is_read_only()
        return "read only"
    monkeypatch.setattr(status_presentation, "format_progress", presentation)
    report = inspect_project(path)
    assert report["sections"]["status_presentation"]["available"]
    assert path.read_bytes() == before
