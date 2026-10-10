from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from phase_agent.graphs.actions.graph import build_tool_action_graph, run_tool_action_graph


def test_parent_retry_reinitializes_runtime_only_action_child():
    calls = []
    fail = [True]
    child = build_tool_action_graph(registry={})
    def execute(state):
        def initialize():
            calls.append("initialize")
            return {"_graph_prepared": True, "current": {"ready": True}}
        def prepare(frame):
            assert frame["current"]["ready"]
            if fail[0]:
                fail[0] = False
                raise RuntimeError("interrupted proposal")
            return {"status": "awaiting_approval"}
        result = run_tool_action_graph(tool_graph=child, registry={}, initialize=initialize,
            refresh=lambda frame: frame, prepare=prepare)
        return {"result": result}
    graph = StateGraph(dict)
    graph.add_node("execute", execute)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume"}}
    import pytest
    with pytest.raises(RuntimeError, match="interrupted proposal"):
        compiled.invoke({}, config)
    assert compiled.invoke(None, config)["result"]["status"] == "awaiting_approval"
    assert calls == ["initialize", "initialize"]
