"""Separate scientific task status from an explicitly waived DFT wait."""

DFT_STAGES = {"dft_single_point", "dft_relax"}


def awaiting_task_result(task):
    return (
        task.get("status") in {"pending", "running", "submitted", "unknown"}
        and not (task.get("model_refresh_id") and task.get("refresh_wait_waived") is True)
        and not (task.get("stage") in DFT_STAGES and task.get("recovery_wait_waived") is True)
    )


def active_pending_tasks(tasks):
    """Project the same unresolved tasks used by the workflow wait boundary."""
    return [task for task in tasks if awaiting_task_result(task)]
