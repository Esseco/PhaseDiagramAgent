import json
import pytest
from langchain_core.messages import HumanMessage
from phase_agent.graphs.studio_chat_graph import build_studio_graph
from phase_agent.graphs.project.runner import run_search_workflow_graph
from phase_agent.graphs.invocation_context import new_invocation


def test_studio_retry_resumes_failed_analysis_without_recollecting(tmp_path, monkeypatch):
    config_path = tmp_path / "agent_runtime.json"
    config_path.write_text(json.dumps({"state_path": "workflow_state/state.json"}))
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config_path))
    calls = []
    failed = [False]
    def analyze(frame):
        if not failed[0]:
            failed[0] = True
            raise RuntimeError("analysis interrupted")
        return frame
    def sender(text, thread):
        result = run_search_workflow_graph(
            checkpoint_path=tmp_path / "workflow_state/langgraph_lifecycle.sqlite",
            project_thread="project-scientific-lifecycle-v2", invocation_id=new_invocation(),
            initialize=lambda: {"_workflow_prepared": True, "snapshot": {"config_version": "v1"}},
            collect=lambda frame: calls.append("collect") or frame,
            analyze=analyze, wait=lambda frame: frame, assess=lambda frame: frame,
            act=lambda frame, **kw: frame, finalize=lambda frame: {"status": "done"})
        return result["status"]
    graph = build_studio_graph(sender, receipt_path=tmp_path / "gateway/receipt.json")
    request = {"messages": [HumanMessage(content="继续", id="m")]}
    config = {"configurable": {"thread_id": "t"}}
    with pytest.raises(RuntimeError, match="analysis interrupted"):
        graph.invoke(request, config)
    result = graph.invoke(request, config)
    assert result["messages"][-1].content == "done"
    assert calls == ["collect"]
    assert "禁止历史重放" in graph.invoke(request, config)["messages"][-1].content
