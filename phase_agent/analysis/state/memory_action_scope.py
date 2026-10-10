"""Choose a retrieval scope from evidence, never from task array order."""


def memory_action_scope(state):
    explicit = state.get("decision_action")
    if explicit:
        return {"action": explicit, "source": "explicit_decision_action"}
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
