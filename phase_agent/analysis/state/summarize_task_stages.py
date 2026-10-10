"""Order-independent task evidence; terminal compute is not accepted recovery."""

from collections import Counter


def summarize_task_stages(state):
    accepted = set(state.get("processed_task_ids") or [])
    groups = {}
    for task in state.get("tasks") or []:
        stage = task.get("stage") or "unknown"
        row = groups.setdefault(
            stage,
            {
                "stage": stage,
                "task_count": 0,
                "status_counts": Counter(),
                "accepted_count": 0,
                "completed_unaccepted_count": 0,
            },
        )
        row["task_count"] += 1
        status = task.get("status") or "unknown"
        row["status_counts"][status] += 1
        is_accepted = bool(task.get("task_id") and task["task_id"] in accepted)
        row["accepted_count"] += is_accepted
        row["completed_unaccepted_count"] += status == "completed" and not is_accepted
    rows = []
    for stage in sorted(groups):
        row = groups[stage]
        counts = row["status_counts"]
        row["status_counts"] = dict(sorted(counts.items()))
        row["failed_or_timeout_count"] = counts.get("failed", 0) + counts.get("timeout", 0)
        row["all_completed_and_accepted"] = (
            counts.get("completed", 0) == row["task_count"]
            and row["accepted_count"] == row["task_count"]
        )
        rows.append(row)
    return {
        "stages": rows,
        "scope": "all_recorded_tasks_not_current_round",
        "instruction": "Task order does not imply stage progress. Completed but unaccepted results need recovery. Use batch/round evidence for next-action decisions; this summary grants no execution authority.",
    }
