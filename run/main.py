"""Single public entry for the resumable Agent event loop."""

from copy import deepcopy
import json
from pathlib import Path

from config_layer.runtime.build_effective_run_config import build_effective_run_config
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from execution_layer.workflows.create_workflow_handlers import create_workflow_handlers
from execution_layer.workflows.run_event_loop import run_event_loop
from execution_layer.workflows.apply_scientific_feedback import apply_scientific_feedback
from execution_layer.state.reconcile_task_results import reconcile_task_results
from config_layer.runtime.authorize_budget_extension import authorize_budget_extension


def run_workflow(
    manager,
    phase_references,
    run_config,
    config_session,
    *,
    state=None,
    handlers=None,
    registry=None,
    agent_client=None,
    execution_mode="autonomous",
    human_feedback=None,
    replay_record=None,
    recovered_results=None,
    max_steps=1,
    invocation_id=None,
    state_path=None,
    approval_directory=None,
    initial_long_term_advice=None,
    dispatcher=None,
    task_runner=None,
    approve_budget_extension=False,
    **runtime_adapters,
):
    """Build the effective config and run bounded validated Agent actions."""
    snapshot = config_session.get("confirmed_snapshot") or {}
    if config_session.get("status") != "confirmed" or not snapshot:
        return {"status": "rejected", "reason": "configuration_not_confirmed", "submitted": False}
    effective_config = build_effective_run_config(config_session, run_config)
    effective_handlers = create_workflow_handlers(
        handlers,
        stage_registry=runtime_adapters.get("stage_registry"),
        candidates_provider=runtime_adapters.get("dft_candidates_provider"),
        qbc_evaluator=runtime_adapters.get("qbc_evaluator"),
    )
    effective_registry = registry or create_tool_registry(effective_handlers)
    context = {
        "manager": manager,
        "phase_references": phase_references,
        "effective_config": deepcopy(effective_config),
        "dispatcher": dispatcher,
        "agent_client": agent_client,
        **runtime_adapters,
    }
    automatic_results = []
    loaded_state = _read_state_for_runner(state, state_path or effective_config.get("state_path"))
    migration = authorize_budget_extension(
        loaded_state, snapshot, user_approved=approve_budget_extension
    )
    if migration["status"].startswith("rejected") or migration["status"] == "approval_required":
        return {"status": "rejected", "reason": migration["status"], "submitted": False,
                "config_version": snapshot["config_version"], "state": loaded_state}
    loaded_state = migration["state"]
    if task_runner is not None:
        automatic_results = task_runner.collect_results(loaded_state)
    combined_results = [*(recovered_results or []), *automatic_results]
    pre_reconciled = reconcile_task_results(loaded_state, combined_results)
    feedback = apply_scientific_feedback(
        pre_reconciled["state"], combined_results, manager=manager,
        ledger_path=effective_config.get("ledger_path"),
        phase_diagram_directory=effective_config.get("phase_diagram_directory"),
        final_frame_mlip_evaluator=runtime_adapters.get("final_frame_mlip_evaluator"),
    )
    feedback["state"]["budget_limits"] = deepcopy(effective_config.get("budgets") or {})
    feedback["state"]["branch_candidates"] = _branch_candidates_for_agent(manager)
    total_limit = (effective_config.get("budgets") or {}).get("total_relative_cost")
    if total_limit is not None:
        used = float((feedback["state"].get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
        reserved = float(feedback["state"].get("reserved_relative_cost", 0) or 0)
        feedback["state"]["budget_remaining"] = max(0.0, float(total_limit) - used - reserved)
    result = run_event_loop(
        feedback["state"],
        config_session,
        registry=effective_registry,
        agent_client=agent_client,
        context=context,
        execution_mode=execution_mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
        recovered_results=None,
        max_steps=max_steps,
        invocation_id=invocation_id,
        state_path=state_path or effective_config.get("state_path"),
        approval_directory=approval_directory or effective_config.get("approval_directory"),
        initial_long_term_advice=initial_long_term_advice,
    )
    result["scientific_feedback"] = {
        key: value for key, value in feedback.items() if key != "state"
    }
    result["reconciled"] = pre_reconciled["reconciled"]
    if task_runner is not None and result["state"].get("pending_tasks"):
        batch_result = task_runner.prepare(result["state"])
        result["state"] = batch_result["state"]
        result["batch"] = batch_result["batch"]
        if batch_result["status"] in {"prepared", "submitted"}:
            result["status"] = f"tasks_{batch_result['status']}"
        _save_runner_state(result["state"], state_path or effective_config.get("state_path"))
    _save_runner_state(result["state"], state_path or effective_config.get("state_path"))
    result.update({
        "submitted": result["status"] not in {"rejected", "rejected_by_user"},
        "config_version": snapshot["config_version"],
        "effective_config": effective_config,
    })
    return result


def _branch_candidates_for_agent(manager):
    """Expose bounded factual branch choices before the Agent proposes a batch."""
    rows = []
    if manager is None:
        return rows
    for branch_id in sorted(manager.data.get("branches", {})):
        branch = manager.data["branches"][branch_id]
        rows.append({key: deepcopy(value) for key, value in {
            "branch_id": branch_id, "P": branch.get("P"), "x": branch.get("x"),
            "T": branch.get("T"), "composition": branch.get("composition"),
            "structure_count": len(branch.get("structure_ids") or []),
        }.items()})
    return rows


def _read_state_for_runner(state, state_path):
    if isinstance(state, (str, Path)):
        return json.loads(Path(state).read_text(encoding="utf-8"))
    if state is not None:
        return deepcopy(state)
    if state_path and Path(state_path).is_file():
        return json.loads(Path(state_path).read_text(encoding="utf-8"))
    return {}


def _save_runner_state(state, state_path):
    if state_path is None:
        return
    output = Path(state_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
