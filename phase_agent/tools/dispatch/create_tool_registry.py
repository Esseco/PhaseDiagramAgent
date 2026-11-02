"""Register safe adapters; handlers reuse existing modules supplied by the caller."""


def create_tool_registry(handlers=None) -> dict:
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog
    from phase_agent.agents.permissions import advisory_roles

    handlers = handlers or {}
    specs = {
        "generate_branches": "generation",
        "select_candidates": "selection",
        "run_calculation_stage": "calculation",
        "allocate_mc_bohb": "bohb",
        "select_dft_candidates": "decision_layer",
        "update_mlip": "scientific_layer",
        "reevaluate_candidates": "analysis_layer",
        "check_convergence": "analysis_layer",
        "pause_search": "execution_layer",
        "restart_failed_task": "state",
        "adjust_strategy": "policy",
        "prepare_dedup_batch": "execution_layer",
        "prepare_local_batch_files": "phase_agent.tools.local",
    }
    child_reservations = {
        "select_dft_candidates",
        "allocate_mc_bohb",
        "prepare_dedup_batch",
        "prepare_local_batch_files",
        "restart_failed_task",
    }
    return {
        name: {
            "name": name,
            "owner": owner,
            "handler": handlers.get(name),
            "proposal_roles": ("strategic",),
            "execution_roles": ("simulation",),
            "advisory_roles": tuple(
                sorted(
                    {
                        role
                        for operation in operation_catalog([name])
                        for role in advisory_roles(operation["operation"])
                    }
                )
            ),
            "parameter_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "failure_policy": "return_to_strategic",
            "operations": operation_catalog([name]) if callable(handlers.get(name)) else [],
            "reservation_mode": "children" if name in child_reservations else "action",
        }
        for name, owner in specs.items()
    }
