"""Public workflow entry: assemble callbacks and invoke the lifecycle graph."""

from functools import partial
from phase_agent.tools.workflows.run_event_loop import run_action_turn
from phase_agent.tools.workflows.lifecycle_initialization import _initialize_workflow
from phase_agent.tools.workflows.lifecycle_recovery import (
    _collect_workflow_results,
    _analyze_workflow_results,
    _workflow_wait_gate,
    _assess_workflow_round,
)
from phase_agent.tools.workflows.lifecycle_finalization import (
    _run_workflow_actions,
    _finalize_workflow,
)


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
    """Run the top-level recovery, analysis, waiting and action StateGraph."""
    if type(max_steps) is not int or max_steps != 1:
        raise ValueError("Scientific workflow permits exactly one action per turn")
    # Dry-run must not reconcile files, activate a model, or invoke task preparers.
    if execution_mode == "dry_run":
        from phase_agent.configuration.runtime.build_effective_run_config import (
            build_effective_run_config,
        )
        from phase_agent.tools.workflows.lifecycle_support import (
            _read_state_for_runner,
            _save_runner_state,
        )
        from phase_agent.tools.workflows.create_workflow_handlers import create_workflow_handlers
        from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
        from phase_agent.tools.policy.workflow_policy import normalize_workflow_state

        config = build_effective_run_config(config_session, run_config)
        path = state_path or config.get("state_path")
        current = normalize_workflow_state(_read_state_for_runner(state, path))
        result = run_action_turn(
            current,
            config_session,
            registry=registry or create_tool_registry(create_workflow_handlers(handlers)),
            agent_client=agent_client,
            execution_mode="dry_run",
            state_path=path,
            context={
                **runtime_adapters,
                "manager": manager,
                "effective_config": config,
                "phase_references": phase_references,
            },
            max_steps=max_steps,
        )
        result["submitted"] = False
        return result
    from phase_agent.graphs.project.runner import run_search_workflow_graph
    from pathlib import Path

    checkpoint_path = (
        Path(state_path or run_config.get("state_path")).resolve().parent
        / "langgraph_lifecycle.sqlite"
        if state_path or run_config.get("state_path")
        else None
    )
    from phase_agent.graphs.invocation_context import lifecycle_request_id

    return run_search_workflow_graph(
        checkpoint_path=checkpoint_path,
        invocation_id=lifecycle_request_id(invocation_id, human_feedback),
        project_thread="project-scientific-lifecycle-v2" if checkpoint_path else None,
        initialize=lambda: _initialize_workflow(
            manager,
            phase_references,
            run_config,
            config_session,
            state=state,
            handlers=handlers,
            registry=registry,
            agent_client=agent_client,
            execution_mode=execution_mode,
            human_feedback=human_feedback,
            replay_record=replay_record,
            recovered_results=recovered_results,
            max_steps=max_steps,
            invocation_id=invocation_id,
            state_path=state_path,
            approval_directory=approval_directory,
            initial_long_term_advice=initial_long_term_advice,
            dispatcher=dispatcher,
            task_runner=task_runner,
            result_collector=result_collector,
            approve_budget_extension=approve_budget_extension,
            **runtime_adapters,
        ),
        collect=_collect_workflow_results,
        analyze=_analyze_workflow_results,
        wait=_workflow_wait_gate,
        assess=_assess_workflow_round,
        act=lambda frame, tool_graph=None: _run_workflow_actions(
            frame,
            event_loop=partial(run_action_turn, tool_graph=tool_graph),
        ),
        finalize=_finalize_workflow,
    )
