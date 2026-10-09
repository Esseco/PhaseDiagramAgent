from copy import deepcopy
from run.flow_panel import render_flow_panel


def test_panel_is_read_only_escaped_and_has_science_and_real_edges():
    state = {"active_model_version": "<script>bad</script>", "branches": [{"branch_id": "b"}],
             "tasks": [{"stage": "deep_search", "status": "completed"}],
             "pending_execution_policies": {"p": {"agent_proposal": {
                 "recommended_action": "update_mlip", "expected_purpose": "<img src=x>"}}}}
    original = deepcopy(state)
    html = render_flow_panel(state)
    assert state == original
    assert "<script>bad" not in html and "<img src=x>" not in html
    assert "&lt;img src=x&gt;" in html
    assert "累计登记 1" in html and "completed: 1" in html
    assert "approval_gate" not in html  # Human-readable adapter label.
    assert "执行审批策略" in html
    assert "tool__update_mlip" in html
    assert "<details>" in html
    assert "不代表本轮全部完成" in html


def test_route_is_get_only():
    from run.studio_runtime import app
    route = next(route for route in app.routes if route.path == "/phase/flow")
    assert "GET" in route.methods and "POST" not in route.methods


def test_endpoint_reads_state_without_workflow(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    from starlette.testclient import TestClient
    from run.chat_application import RunWorkflowChatHandler
    from run import studio_runtime
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"active_model_version": "m1"}), encoding="utf-8")
    before = path.read_bytes()
    workflow = Mock()
    handler = RunWorkflowChatHandler({"state_path": str(path)}, workflow=workflow)
    monkeypatch.setattr(studio_runtime, "_handler", handler)
    client = TestClient(studio_runtime.app)  # Do not start real service lifespan.
    response = client.get("/phase/flow")
    assert response.status_code == 200
    assert "科学流程" in response.text
    assert response.headers["cache-control"] == "no-store"
    assert client.post("/phase/flow").status_code == 405
    workflow.assert_not_called()
    assert path.read_bytes() == before
