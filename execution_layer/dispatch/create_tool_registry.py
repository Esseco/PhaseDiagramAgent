"""Register safe adapters; handlers reuse existing modules supplied by the caller."""


def create_tool_registry(handlers=None) -> dict:
    handlers = handlers or {}
    specs = {
        "generate_branches": "generation", "select_candidates": "selection", "run_calculation_stage": "calculation",
        "allocate_mc_bohb": "bohb", "select_dft_candidates": "decision_layer", "update_mlip": "scientific_layer",
        "reevaluate_candidates": "analysis_layer", "check_convergence": "analysis_layer", "pause_search": "execution_layer",
        "restart_failed_task": "state", "adjust_strategy": "policy",
        "prepare_dedup_batch": "execution_layer",
        "prepare_local_batch_files": "execution_layer.local",
    }
    child_reservations = {"select_dft_candidates", "allocate_mc_bohb", "prepare_dedup_batch",
                          "restart_failed_task"}
    return {
        name: {
            "name": name,
            "owner": owner,
            "handler": handlers.get(name),
            "reservation_mode": "children" if name in child_reservations else "action",
        }
        for name, owner in specs.items()
    }
