import pytest
from phase_agent.graphs.actions.graph import build_tool_action_graph, run_tool_action_graph


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("outcome", ["approve", "wait", "invalid"])
def test_grouping_preserves_gates_and_single_dispatch(grouped, outcome):
    calls = []
    def approve(frame):
        calls.append("approve")
        return {"status": "awaiting_approval"} if outcome == "wait" else frame
    def validate(frame):
        calls.append("validate")
        return {**frame, "action": {"tool": "update_mlip"}, "dispatch_ready": outcome != "invalid"}
    def dispatch(frame):
        calls.append("dispatch")
        return frame
    def finish(frame):
        calls.append("audit")
        return {"status": "completed" if outcome == "approve" else "rejected"}
    registry = {"update_mlip": {}, "allocate_mc_bohb": {}, "custom": {}}
    graph = build_tool_action_graph(registry=registry, group_actions=grouped)
    result = run_tool_action_graph(tool_graph=graph, registry=registry,
        initialize=lambda: {"_graph_prepared": True}, refresh=lambda f: f, prepare=lambda f: f,
        approve=approve, validate=validate, dispatch=dispatch, finish=finish)
    assert calls == ({"approve": ["approve", "validate", "dispatch", "audit"],
                      "wait": ["approve"], "invalid": ["approve", "validate", "audit"]}[outcome])
    assert result["status"] == {"approve": "completed", "wait": "awaiting_approval", "invalid": "rejected"}[outcome]
