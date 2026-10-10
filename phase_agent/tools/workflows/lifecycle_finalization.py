"""Action invocation and prepared-input finalization."""

from phase_agent.decisions.agent.dialogue_contract import DIALOGUE_STATUSES
from copy import deepcopy
from phase_agent.tools.workflows.lifecycle_support import _save_runner_state


def prepare_workflow_actions(frame, *, event_loop):
    feedback = frame["feedback"]
    config_session = frame["config_session"]
    effective_registry = frame["effective_registry"]
    agent_client = frame["agent_client"]
    context = frame["context"]
    execution_mode = frame["execution_mode"]
    human_feedback = frame["human_feedback"]
    replay_record = frame["replay_record"]
    max_steps = frame["max_steps"]
    invocation_id = frame["invocation_id"]
    state_path = frame["state_path"]
    effective_config = frame["effective_config"]
    approval_directory = frame["approval_directory"]
    initial_long_term_advice = frame["initial_long_term_advice"]
    return event_loop(
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


def complete_workflow_actions(frame, result):
    result["scientific_feedback"] = {
        key: value for key, value in frame["feedback"].items() if key != "state"
    }
    result["reconciled"] = frame["pre_reconciled"]["reconciled"]
    return {
        **frame,
        "result": result,
    }


def _run_workflow_actions(frame, *, event_loop):
    from phase_agent.tools.workflows.propose_followup import propose_after_generation

    result = prepare_workflow_actions(frame, event_loop=event_loop)
    result = propose_after_generation(frame, result, event_loop=event_loop)
    return complete_workflow_actions(frame, result)


def _finalize_workflow(frame):
    from phase_agent.tools.workflows.model_refresh_state import refresh_active
    from phase_agent.tools.remote.summarize_manual_upload_wait import summarize_manual_upload_wait

    task_runner = frame["task_runner"]
    result = frame["result"]
    state_path = frame["state_path"]
    effective_config = frame["effective_config"]
    recovered_count = frame["recovered_count"]
    execution_mode = frame["execution_mode"]
    collection_report = frame["collection_report"]
    snapshot = frame["snapshot"]
    action_executed = any(
        (event.get("execution") or {}).get("status") == "completed"
        for event in result.get("events") or []
    )
    if (
        action_executed
        and task_runner is not None
        and result["state"].get("pending_tasks")
        and not refresh_active(result["state"])
    ):
        batch_result = task_runner.prepare(result["state"])
        result["state"] = batch_result["state"]
        result["batch"] = {**batch_result["batch"], "status": batch_result["status"]}
        if batch_result["status"] in {"prepared", "submitted"}:
            result["status"] = f"tasks_{batch_result['status']}"
        _save_runner_state(result["state"], state_path or effective_config.get("state_path"))
    manual_wait = (
        summarize_manual_upload_wait(result["state"], recovered_count=recovered_count)
        if execution_mode == "interactive"
        else None
    )
    if manual_wait is not None and collection_report is not None:
        manual_wait["result_collection"] = deepcopy(collection_report)
    if manual_wait and result.get("status") not in DIALOGUE_STATUSES | {
        "awaiting_approval",
        "rejected",
    }:
        result["status"] = "awaiting_manual_submission"
        result["manual_wait"] = manual_wait
    result["recovered_count"] = recovered_count
    from phase_agent.analysis.feedback.export_dft_products import export_dft_products

    export_dft_products(result["state"], effective_config.get("phase_diagram_directory"))
    _save_runner_state(result["state"], state_path or effective_config.get("state_path"))
    result.update(
        {
            "submitted": (result.get("batch") or {}).get("status") == "submitted"
            or any(
                ((event.get("execution") or {}).get("result") or {}).get("submitted") is True
                for event in result.get("events") or []
            ),
            "config_version": snapshot["config_version"],
            "effective_config": effective_config,
            "result_collection": collection_report,
        }
    )
    if result["state"].get("model_refresh"):
        result["submitted"] = False  # This workflow prepares manual HPC inputs only.
    return result
