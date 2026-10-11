import pytest

from phase_agent.graphs.project.runner import run_search_workflow_graph


def test_new_request_at_wait_uses_new_configuration_once(tmp_path):
    version = ["v1"]
    calls = []

    def initialize():
        calls.append(("initialize", version[0]))
        return {"_workflow_prepared": True, "snapshot": {"config_version": version[0]}}

    def collect(frame):
        calls.append(("collect", frame["snapshot"]["config_version"]))
        return frame

    stages = dict(
        initialize=initialize,
        collect=collect,
        analyze=lambda f: f,
        wait=lambda f: {"status": "awaiting_approval"},
        assess=lambda f: f,
        act=lambda f, **kw: pytest.fail("no scientific action"),
        finalize=lambda f: f,
    )
    options = dict(
        checkpoint_path=tmp_path / "lifecycle.sqlite", project_thread="project", **stages
    )
    assert (
        run_search_workflow_graph(invocation_id="old", **options)["status"] == "awaiting_approval"
    )
    version[0] = "v2"
    # The old message is cached, not re-executed under a new config.
    assert (
        run_search_workflow_graph(invocation_id="old", **options)["status"] == "awaiting_approval"
    )
    assert calls == [("initialize", "v1"), ("collect", "v1")]
    assert (
        run_search_workflow_graph(invocation_id="new", **options)["status"] == "awaiting_approval"
    )
    assert calls == [
        ("initialize", "v1"),
        ("collect", "v1"),
        ("initialize", "v2"),
        ("collect", "v2"),
    ]
    assert (
        run_search_workflow_graph(invocation_id="new", **options)["status"] == "awaiting_approval"
    )
    assert len(calls) == 4


@pytest.mark.parametrize("new_request", [False, True])
def test_config_change_still_rejects_failed_inflight_checkpoint(tmp_path, new_request):
    version = ["v1"]
    calls = []

    def initialize():
        return {"_workflow_prepared": True, "snapshot": {"config_version": version[0]}}

    def analyze(frame):
        calls.append("analyze")
        raise RuntimeError("failed before action")

    options = dict(
        checkpoint_path=tmp_path / "lifecycle.sqlite",
        project_thread="project",
        initialize=initialize,
        collect=lambda f: f,
        analyze=analyze,
        wait=lambda f: f,
        assess=lambda f: f,
        act=lambda f, **kw: pytest.fail("no action"),
        finalize=lambda f: f,
    )
    with pytest.raises(RuntimeError, match="failed before action"):
        run_search_workflow_graph(invocation_id="old", **options)
    version[0] = "v2"
    with pytest.raises(ValueError, match="Lifecycle configuration changed"):
        run_search_workflow_graph(invocation_id="new" if new_request else "old", **options)
    assert calls == ["analyze"]
