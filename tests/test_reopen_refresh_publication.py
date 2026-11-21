import json

import pytest

from phase_agent.tools.workflows.reopen_refresh_publication import commit_reopen


@pytest.mark.parametrize("fail_after_write", [False, True])
def test_status_and_state_failure_rollback(tmp_path, monkeypatch, fail_after_write):
    from phase_agent.analysis.feedback import versioned_products
    from phase_agent.persistence.state import read_versions
    from phase_agent.tools.workflows import lifecycle_support

    root = tmp_path / "analysis_outputs"
    state_path = tmp_path / "workflow_state" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text("{}")
    latest = root / "latest" / "phase_diagram_status.json"
    epoch = root / "epoch2" / "phase_diagram" / "latest_status.json"
    for path in (latest, epoch):
        path.parent.mkdir(parents=True)
        path.write_text('{"status":"completed"}')
    def publish(root, kind, model, epoch_name, **kw):
        for path in (latest, epoch):
            path.write_text(json.dumps(kw))
    def save(state, path, *, expected_identity):
        if fail_after_write:
            path.write_text('{"committed":true}')
        raise OSError("simulated")
    monkeypatch.setattr(versioned_products, "publication_status", publish)
    monkeypatch.setattr(lifecycle_support, "_save_runner_state", save)
    monkeypatch.setattr(read_versions, "read_identity", lambda *args, **kw: "after" if fail_after_write else "before")
    with pytest.raises(OSError):
        commit_reopen({"analysis_output_directory": str(root), "active_model_version": "new"}, state_path, "before", epoch="epoch2")
    for path in (latest, epoch):
        assert json.loads(path.read_text())["status"] == ("not_ready" if fail_after_write else "completed")


def test_success_publishes_pending_and_commits_once(tmp_path, monkeypatch):
    from phase_agent.analysis.feedback import versioned_products
    from phase_agent.tools.workflows import lifecycle_support
    root = tmp_path / "analysis_outputs"
    state_path = tmp_path / "workflow_state" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text("{}")
    calls = []
    def publish(root_arg, kind, model, epoch, **kwargs):
        calls.append(("publish", kwargs["status"]))
    def save(state, path, *, expected_identity):
        calls.append(("save", expected_identity))
    monkeypatch.setattr(versioned_products, "publication_status", publish)
    monkeypatch.setattr(lifecycle_support, "_save_runner_state", save)
    commit_reopen({"analysis_output_directory": str(root), "active_model_version": "new"}, state_path, "before", epoch="epoch2")
    assert calls == [("publish", "not_ready"), ("save", "before")]
