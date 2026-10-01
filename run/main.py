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
from execution_layer.step_runner.file_protocol import write_json
from execution_layer.workflows.compact_action_history import compact_state_history


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
    execution_mode="interactive",
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
    result_collector=None,
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
    collection_report = None
    loaded_state = compact_state_history(
        _read_state_for_runner(state, state_path or effective_config.get("state_path"))
    )
    migration = authorize_budget_extension(
        loaded_state, snapshot, user_approved=approve_budget_extension
    )
    if migration["status"].startswith("rejected") or migration["status"] == "approval_required":
        return {"status": "rejected", "reason": migration["status"], "submitted": False,
                "config_version": snapshot["config_version"], "state": loaded_state,
                "migration": migration,
                "message": _config_migration_block_message(migration)}
    loaded_state = migration["state"]
    if approve_budget_extension:
        if migration["status"] in {"extended", "empty_run_rebound"}:
            limits = deepcopy(effective_config.get("budgets") or {})
            loaded_state["budget_limits"] = limits
            total_limit = limits.get("total_relative_cost")
            if total_limit is not None:
                used = float((loaded_state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
                reserved = float(loaded_state.get("reserved_relative_cost", 0) or 0)
                loaded_state["budget_remaining"] = max(0.0, float(total_limit) - used - reserved)
            _save_runner_state(loaded_state, state_path or effective_config.get("state_path"))
            return {
                "status": "config_migrated", "state": loaded_state,
                "migration": {key: value for key, value in migration.items() if key != "state"},
                "from_config_version": migration.get("from"),
                "config_version": snapshot["config_version"],
                "submitted": False,
            }
        if migration["status"] == "unchanged":
            return {
                "status": "config_migration_not_needed", "state": loaded_state,
                "config_version": snapshot["config_version"], "submitted": False,
            }
    from execution_layer.state.restore_generation_gate import restore_generation_gate
    loaded_state = restore_generation_gate(loaded_state, manager)
    collector = task_runner or result_collector
    if collector is not None:
        if hasattr(collector, "collect_results_with_report"):
            collected = collector.collect_results_with_report(loaded_state)
            automatic_results = collected["results"]
            collection_report = collected["report"]
        else:
            automatic_results = collector.collect_results(loaded_state)
    combined_results = [*(recovered_results or []), *automatic_results]
    pre_reconciled = reconcile_task_results(loaded_state, combined_results)
    runtime_state_path = state_path or effective_config.get("state_path")
    phase_cache_path = runtime_adapters.get("phase_identification_cache_path")
    if phase_cache_path is None and runtime_state_path:
        phase_cache_path = Path(runtime_state_path).with_name("phase_identification_cache.json")
    feedback = apply_scientific_feedback(
        pre_reconciled["state"], combined_results, manager=manager,
        ledger_path=effective_config.get("ledger_path"),
        phase_diagram_directory=effective_config.get("phase_diagram_directory"),
        final_frame_mlip_evaluator=_dft_comparison_evaluator(runtime_adapters, effective_config),
        active_model_version=(effective_config.get("mlip") or {}).get("version")
            or (effective_config.get("mlip") or {}).get("name"),
        phase_references=phase_references,
        phase_identification_cache_path=phase_cache_path,
    )
    from analysis_layer.phase.update_local_mlip_hull_pool import update_local_mlip_hull_pool
    feedback["state"] = update_local_mlip_hull_pool(
        feedback["state"], config=effective_config,
        path=effective_config.get("branch_energy_pool_ledger_path"))
    feedback["state"]["budget_limits"] = deepcopy(effective_config.get("budgets") or {})
    feedback["state"]["branch_candidates"] = _branch_candidates_for_agent(manager)
    total_limit = (effective_config.get("budgets") or {}).get("total_relative_cost")
    if total_limit is not None:
        used = float((feedback["state"].get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
        reserved = float(feedback["state"].get("reserved_relative_cost", 0) or 0)
        feedback["state"]["budget_remaining"] = max(0.0, float(total_limit) - used - reserved)
    recovered_count = _count_newly_recovered(pre_reconciled["reconciled"])
    from execution_layer.remote.summarize_manual_upload_wait import summarize_manual_upload_wait
    manual_wait = (summarize_manual_upload_wait(feedback["state"], recovered_count=recovered_count)
                   if execution_mode == "interactive" else None)
    if manual_wait is not None and collection_report is not None:
        manual_wait["result_collection"] = deepcopy(collection_report)
    from execution_layer.local.rebuild_relax_inputs import is_relax_rebuild_request
    rebuilding = is_relax_rebuild_request(runtime_adapters.get("user_message")) or any(
        (((row.get("agent_proposal") or {}).get("raw_action") or {}).get("parameters") or {}).get("rebuild_inputs")
        for row in feedback["state"].get("pending_execution_policies", {}).values())
    if manual_wait and not rebuilding:
        wait_state = feedback["state"]
        _save_runner_state(wait_state, state_path or effective_config.get("state_path"))
        return {
            "status": "awaiting_manual_submission", "state": wait_state,
            "manual_wait": manual_wait, "recovered_count": recovered_count,
            "scientific_feedback": {key: value for key, value in feedback.items() if key != "state"},
            "reconciled": pre_reconciled["reconciled"], "steps_executed": 0,
            "submitted": False, "config_version": snapshot["config_version"],
            "effective_config": effective_config, "result_collection": collection_report,
        }
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
    manual_wait = (summarize_manual_upload_wait(result["state"], recovered_count=recovered_count)
                   if execution_mode == "interactive" else None)
    if manual_wait is not None and collection_report is not None:
        manual_wait["result_collection"] = deepcopy(collection_report)
    if manual_wait:
        result["status"] = "awaiting_manual_submission"
        result["manual_wait"] = manual_wait
    result["recovered_count"] = recovered_count
    _save_runner_state(result["state"], state_path or effective_config.get("state_path"))
    result.update({
        "submitted": result["status"] not in {"rejected", "rejected_by_user", "awaiting_manual_submission"},
        "config_version": snapshot["config_version"],
        "effective_config": effective_config,
        "result_collection": collection_report,
    })
    return result


def _config_migration_block_message(migration):
    status = migration.get("status")
    source, target = migration.get("from") or "未知", migration.get("to") or "未知"
    if status == "approval_required":
        return (
            f"运行仍绑定配置 {source}，当前确认配置为 {target}；已有任务或结果。"
            "若要迁移预算上限、已确认开启第二段 MC，或调整 MC 步数成本估算系数，"
            "请核对配置后发送“批准迁移”；其他科学设置变化不会被迁移。"
        )
    if status == "rejected_non_budget_change":
        fields = "、".join(migration.get("changed_fields") or []) or "科学设置"
        evidence = migration.get("compatibility_rejection")
        evidence_text = f"兼容性核验：{evidence}。" if evidence else ""
        return (
            f"已收到迁移批准，但配置 {source} → {target} 改动了 {fields}。"
            f"{evidence_text}运行状态和历史结果未修改。"
        )
    if status == "rejected_budget_decrease":
        return (
            f"配置 {source} → {target} 降低了已有运行的预算上限，不能迁移；运行状态和历史结果未修改。"
        )
    return f"配置迁移被拒绝（{status}）；运行状态和历史结果未修改。"


def _count_newly_recovered(reconciled):
    return sum(row.get("status") in {"settled", "already_settled"}
               for row in (reconciled or []))


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
    write_json(state_path, state)


def _dft_comparison_evaluator(adapters, config):
    if "final_frame_mlip_evaluator" in adapters:
        return adapters["final_frame_mlip_evaluator"]
    from execution_layer.workflows.create_dft_comparison_evaluator import create_dft_comparison_evaluator
    return create_dft_comparison_evaluator(config)
