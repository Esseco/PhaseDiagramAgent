"""Withdrawn generation batches remain audit facts, never active candidate pools."""


def retirement_records(state):
    return (state.get("branch_region_review") or {}).get("retired_batches") or []


def retired_task_keys(state):
    return {row["task_key"] for row in retirement_records(state)}


def retired_branch_ids(state):
    records = retirement_records(state)
    retired = {bid for row in records for bid in row.get("branch_ids") or []}
    keys = retired_task_keys(state)
    # A later, explicitly approved generation may reproduce the same stable ID.
    registered_again = {
        bid
        for row in state.get("generation_history") or []
        if row.get("task_key") not in keys
        for bid in row.get("new_branch_ids") or []
    }
    return retired - registered_again
