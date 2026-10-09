"""Public workflow entry: assemble callbacks and invoke the lifecycle graph."""
from execution_layer.workflows.run_event_loop import run_event_loop
from execution_layer.workflows.lifecycle_initialization import _initialize_workflow
from execution_layer.workflows.lifecycle_recovery import (
    _collect_workflow_results, _analyze_workflow_results,
    _workflow_wait_gate, _assess_workflow_round,
)
from execution_layer.workflows.lifecycle_finalization import _run_workflow_actions, _finalize_workflow


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
    from orchestration.search_workflow_graph import run_search_workflow_graph
    return run_search_workflow_graph(
        initialize=lambda: _initialize_workflow(
            manager, phase_references, run_config, config_session,
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
            approve_budget_extension=approve_budget_extension, **runtime_adapters),
        collect=_collect_workflow_results, analyze=_analyze_workflow_results,
        wait=_workflow_wait_gate, assess=_assess_workflow_round,
        act=lambda frame, event_graph=None: _run_workflow_actions(
            frame, event_loop=lambda *args, **kwargs: run_event_loop(
                *args, **kwargs, event_graph=event_graph)), finalize=_finalize_workflow,
    )
