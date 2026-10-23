"""Count current-round DFT tasks independently of advisory cost budgets."""


def dft_task_limit(config):
    selection = (config.get("dft") or {}).get("selection") or {}
    limit = selection.get("max_tasks_per_round", 100)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("dft.selection.max_tasks_per_round must be an integer in [1, 100]")
    return limit


def round_dft_tasks(state, config):
    model = config.get("mlip") or {}
    version = state.get("active_model_version") or model.get("version") or model.get("name")
    generation_cycle = len(state.get("generation_history") or [])
    return [
        row
        for row in state.get("tasks") or []
        if row.get("stage") in {"dft_single_point", "dft_relax"}
        and row.get("model_version") == version
        and row.get("generation_cycle") == generation_cycle
        and row.get("status") != "cancelled"
    ]
