import pytest
from phase_agent.graphs.project.runner import run_search_workflow_graph


def test_restart_skips_completed_collection_and_completed_invocation(tmp_path):
    calls = []
    fail = [True]
    def initialize():
        return {"_workflow_prepared": True, "snapshot": {"config_version": "v1"},
                "loaded_state": {"collected": False}, "runtime_dependency": object()}
    def collect(frame):
        calls.append("collect")
        return {**frame, "loaded_state": {"collected": True}}
    def analyze(frame):
        assert frame["loaded_state"]["collected"]
        assert "runtime_dependency" in frame
        if fail[0]:
            fail[0] = False
            raise RuntimeError("simulated_restart")
        return frame
    stages = dict(initialize=initialize, collect=collect, analyze=analyze,
        wait=lambda frame: frame, assess=lambda frame: frame, act=lambda frame, **kw: frame,
        finalize=lambda frame: {"status": "done", "state": frame["loaded_state"]})
    kwargs = dict(checkpoint_path=tmp_path / "lifecycle.sqlite", invocation_id="request", **stages)
    with pytest.raises(RuntimeError, match="simulated_restart"):
        run_search_workflow_graph(**kwargs)
    result = run_search_workflow_graph(**kwargs)
    assert result == {"status": "done", "state": {"collected": True}}
    assert calls == ["collect"]
    assert run_search_workflow_graph(**kwargs) == result
    assert calls == ["collect"]


def test_new_message_resumes_native_project_wait_with_fresh_records(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from phase_agent.graphs.scientific_graph import scientific_graph
    calls = []
    ready = [False]
    def initialize():
        return {"_workflow_prepared": True, "snapshot": {"config_version": "v1"},
                "loaded_state": {"ready": ready[0]}}
    def collect(frame):
        calls.append(frame["loaded_state"]["ready"])
        return frame
    def wait(frame):
        return frame if frame["loaded_state"]["ready"] else {"status": "awaiting_approval"}
    stages = dict(initialize=initialize, collect=collect, analyze=lambda frame: frame,
        wait=wait, assess=lambda frame: frame, act=lambda frame, **kw: frame,
        finalize=lambda frame: {"status": "done"})
    path = tmp_path / "lifecycle.sqlite"
    kwargs = dict(checkpoint_path=path, project_thread="project", **stages)
    assert run_search_workflow_graph(invocation_id="first", **kwargs)["status"] == "awaiting_approval"
    with SqliteSaver.from_conn_string(str(path)) as saver:
        from copy import copy
        graph = copy(scientific_graph()); graph.checkpointer = saver
        checkpoint = graph.get_state({"configurable": {"thread_id": "project"}})
        assert checkpoint.next == ("wait_for_project_input",)
        assert any(task.interrupts for task in checkpoint.tasks)
    assert run_search_workflow_graph(invocation_id="first", **kwargs)["status"] == "awaiting_approval"
    assert calls == [False]
    ready[0] = True
    assert run_search_workflow_graph(invocation_id="second", **kwargs)["status"] == "done"
    assert calls == [False, True]
    assert run_search_workflow_graph(invocation_id="second", **kwargs)["status"] == "done"
    assert calls == [False, True]
