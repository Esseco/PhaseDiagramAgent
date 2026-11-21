"""Choose a retrieval scope from evidence, never from task array order."""


def memory_action_scope(state):
    explicit = state.get("decision_action")
    if explicit:
        return {"action": explicit, "source": "explicit_decision_action"}
    remote_sampling = state.get("remote_dft_sampling") or {}
    if any(
        isinstance(row, dict) and row.get("status") == "accepted"
        for row in remote_sampling.values()
    ) and not any(
        t.get("stage") in {"dft_single_point", "dft_relax"}
        and t.get("status") in {"pending", "running", "submitted"}
        for t in state.get("tasks") or []
    ):
        return {
            "action": "select_dft_candidates",
            "source": "accepted_remote_dft_sampling",
            "candidate_actions": ["select_dft_candidates"],
            "unknown_stage_present": False,
            "instruction": "Memory retrieval scope only, not a next-action recommendation or round completion proof.",
        }
    active_tasks = [
        t
        for t in state.get("tasks") or []
        if t.get("status") in {"pending", "running", "submitted"}
    ]
    if active_tasks:
        active_candidates = set()
        for task in active_tasks:
            stage = task.get("stage")
            if stage in {"relax", "relax_and_feature", "deep_search"}:
                active_candidates.add("allocate_mc_bohb")
            elif stage in {"dft_single_point", "dft_relax"}:
                active_candidates.add("select_dft_candidates")
        if len(active_candidates) == 1:
            action = next(iter(active_candidates))
            return {
                "action": action,
                "source": "active_task_scope",
                "candidate_actions": [action],
                "unknown_stage_present": False,
                "instruction": "Memory retrieval scope only, not a next-action recommendation or round completion proof.",
            }
    candidates = set()
    unknown = False
    for task in state.get("tasks") or []:
        stage = task.get("stage")
        if stage in {"relax", "relax_and_feature"}:
            candidates.add("allocate_mc_bohb")
        elif stage == "deep_search":
            candidates.add(
                "select_dft_candidates" if task.get("status") == "completed" else "allocate_mc_bohb"
            )
        elif stage in {"dft_single_point", "dft_relax"}:
            candidates.add("select_dft_candidates")
        else:
            unknown = True
    action = next(iter(candidates)) if len(candidates) == 1 and not unknown else None
    return {
        "action": action,
        "source": "unique_task_set_scope" if action else "undetermined_task_set_scope",
        "candidate_actions": sorted(candidates),
        "unknown_stage_present": unknown,
        "instruction": "Memory retrieval scope only, not a next-action recommendation or round completion proof.",
    }
