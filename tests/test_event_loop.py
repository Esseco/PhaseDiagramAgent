import json

from config_layer.runtime.build_effective_run_config import build_effective_run_config
from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.create_config_draft import create_config_draft
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from execution_layer.workflows.create_workflow_handlers import create_workflow_handlers
from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.budget.reserve_action_budget import reserve_action_budget
from execution_layer.workflows.run_event_loop import run_event_loop
from run import run_workflow


def _session():
    draft = create_config_draft(default_layered_search_config())
    draft = apply_config_revision(draft, {
        "calculation.mlip_version": "m1",
        "dft.parameters": {"encut": 520},
        "convergence.hull_tolerance": 0.01,
        "convergence.minimum_dft_validations": 1,
        "convergence.coverage_threshold": 0.9,
        "mlip.model_path": "/remote/mace-mh-1.model",
    })
    return confirm_config_snapshot(draft, user_confirmed=True)


def test_confirmed_config_overrides_protected_runtime_values():
    session = _session()
    effective = build_effective_run_config(
        session,
        {
            "state_path": "state.json",
            "seed": 7,
            "dft": {"parameters": {"encut": 1}},
            "budgets": {"total_relative_cost": 0},
            "mlip": {"name": "mace-mh-1", "model_path": None, "model_paths": [],
                     "device": "cuda"},
            "unknown": "ignored",
        },
    )
    confirmed = session["confirmed_snapshot"]["config"]
    assert effective["dft"] == confirmed["dft"]
    assert effective["budgets"] == confirmed["budgets"]
    assert effective["seed"] == 42
    assert effective["mlip"]["model_path"] == "/remote/mace-mh-1.model"
    assert effective["mlip"]["device"] == "cuda"
    assert effective["ignored_runtime_keys"] == ["budgets", "dft", "seed", "unknown"]


def test_runtime_relax_defaults_do_not_change_confirmed_result_identity():
    session = {"status": "confirmed", "confirmed_snapshot": {
        "config_version": "v1", "config": {"mlip": {"name": "m1"},
            "calculation": {"mlip_version": "m1"}, "run": {}}}}
    effective = build_effective_run_config(session, {
        "mlip": {"relax_parameters": {"fmax": 0.05, "relax_steps": 150}},
    })
    assert "relax_parameters" not in effective["mlip"]


def test_autonomous_event_loop_routes_every_action_through_policy_and_registry():
    session = _session()
    actions = iter([
        {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "inspect"},
        {"tool": "pause_search", "parameters": {}, "budget": 0, "reason": "pause"},
    ])
    registry = create_tool_registry(create_workflow_handlers())
    result = run_event_loop(
        {}, session, registry=registry,
        agent_client=lambda _: next(actions),
        execution_mode="autonomous", max_steps=2,
    )
    assert result["status"] == "paused"
    assert result["steps_executed"] == 2
    assert all(item["validation"]["valid"] for item in result["events"])
    assert [item["final_action"]["tool"] for item in result["events"]] == ["check_convergence", "pause_search"]
    assert len(result["state"]["action_records"]) == 2


def test_interactive_event_loop_resumes_from_disk(tmp_path):
    session = _session()
    path = tmp_path / "event-state.json"
    first = run_event_loop(
        {}, session,
        registry=create_tool_registry(create_workflow_handlers()),
        agent_client=lambda _: {"tool": "pause_search", "parameters": {}, "budget": 0, "reason": "debug"},
        execution_mode="interactive", invocation_id="approval-1",
        state_path=path,
    )
    assert first["status"] == "awaiting_approval"
    second = run_event_loop(
        path, session,
        registry=create_tool_registry(create_workflow_handlers()),
        execution_mode="interactive", invocation_id="approval-1",
        human_feedback="approve", state_path=path,
    )
    assert second["status"] == "paused"
    assert json.loads(path.read_text(encoding="utf-8"))["action_records"][0]["human_feedback"]["decision"] == "approve"


def test_file_approval_supports_comment_revision_then_explicit_approval(tmp_path):
    session = _session(); state_path = tmp_path / "state.json"; approvals = tmp_path / "approvals"
    registry = create_tool_registry(create_workflow_handlers())
    first = run_event_loop({}, session, registry=registry, agent_client=lambda _: {"tool": "pause_search", "parameters": {}, "budget": 0, "reason": "initial"}, execution_mode="interactive", invocation_id="file-approval", state_path=state_path, approval_directory=approvals)
    action_dir = approvals / "file-approval"
    assert first["status"] == "awaiting_approval"
    assert (action_dir / "proposal-r000.md").exists()
    decision = json.loads((action_dir / "decision-r000.json").read_text(encoding="utf-8"))
    decision.update({"decision": "comment", "comment": "请解释后再暂停"})
    (action_dir / "decision-r000.json").write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")
    revised = run_event_loop(state_path, session, registry=registry, agent_client=lambda payload: {"tool": "pause_search", "parameters": {}, "budget": 0, "reason": f"已分析：{payload['human_comment']}"}, execution_mode="interactive", invocation_id="file-approval", state_path=state_path, approval_directory=approvals)
    assert revised["status"] == "awaiting_approval"
    assert (action_dir / "proposal-r001.json").exists()
    approval = json.loads((action_dir / "decision-r001.json").read_text(encoding="utf-8"))
    approval.update({"decision": "approve", "comment": "同意"})
    (action_dir / "decision-r001.json").write_text(json.dumps(approval, ensure_ascii=False), encoding="utf-8")
    completed = run_event_loop(state_path, session, registry=registry, execution_mode="interactive", invocation_id="file-approval", state_path=state_path, approval_directory=approvals)
    assert completed["status"] == "paused"


def test_recovered_task_settles_once_and_releases_reservation():
    action = {"tool": "run_calculation_stage", "task_key": "K1", "stage": "deep_search", "budget": 4}
    state = reserve_action_budget({}, action, config_version="c1")
    state["tasks"] = [{"task_id": "T1", "task_key": "K1", "status": "pending"}]
    recovered = [{"task_id": "T1", "task_key": "K1", "status": "completed", "actual_cost": 3}]
    first = reconcile_task_results(state, recovered)
    second = reconcile_task_results(first["state"], recovered)
    assert first["state"]["budget_usage"]["total_relative_cost"] == 3
    assert first["state"]["reserved_relative_cost"] == 0
    assert second["reconciled"][0]["status"] == "already_processed"
    assert second["state"]["budget_usage"]["total_relative_cost"] == 3


def test_public_workflow_uses_event_loop_and_effective_config(tmp_path):
    session = _session()
    result = run_workflow(
        None, {}, {"state_path": str(tmp_path / "state.json"), "seed": 3}, session,
        agent_client=lambda _: {"tool": "pause_search", "parameters": {}, "budget": 0, "reason": "done"},
        max_steps=3,
    )
    assert result["status"] == "paused"
    assert result["steps_executed"] == 1
    assert result["effective_config"]["seed"] == 42
    assert "seed" in result["effective_config"]["ignored_runtime_keys"]
    assert result["config_version"] == session["confirmed_snapshot"]["config_version"]


def test_pending_task_is_recovered_and_settled_before_next_agent_action():
    session = _session()
    handlers = create_workflow_handlers({
        "run_calculation_stage": lambda **_: {
            "status": "pending", "task_id": "T1", "task_key": "K1",
        }
    })
    first = run_event_loop(
        {}, session, registry=create_tool_registry(handlers),
        agent_client=lambda _: {
            "tool": "run_calculation_stage", "stage": "deep_search",
            "task_key": "K1", "parameters": {}, "budget": 4,
            "reason": "search",
        },
        execution_mode="autonomous", max_steps=5,
    )
    assert first["status"] == "pending"
    assert first["state"]["budget_reservations"]["K1"]["status"] == "submitted"
    second = run_event_loop(
        first["state"], session, registry=create_tool_registry(create_workflow_handlers()),
        agent_client=lambda _: {
            "tool": "pause_search", "parameters": {}, "budget": 0,
            "reason": "review completed task",
        },
        recovered_results=[{
            "task_id": "T1", "task_key": "K1", "status": "completed",
            "actual_cost": 3,
        }],
        execution_mode="autonomous", max_steps=2,
    )
    assert second["status"] == "paused"
    assert second["state"]["budget_reservations"]["K1"]["status"] == "settled"
    assert second["state"]["budget_usage"]["total_relative_cost"] == 3
    assert second["state"]["reserved_relative_cost"] == 0
