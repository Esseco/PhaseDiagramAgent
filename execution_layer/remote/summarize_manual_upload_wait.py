"""Summarize prepared tasks that still need manual supercomputer execution."""

from pathlib import Path


ACTIVE_STATUSES = {"pending", "running"}


def summarize_manual_upload_wait(state, *, recovered_count=0):
    """Return a compact handoff summary, or None when no prepared task is waiting."""
    tasks = [
        row for row in (state or {}).get("tasks", [])
        if row.get("stage") in {"relax_and_feature", "deep_search",
                                "dft_single_point", "dft_relax"}
        and row.get("batch_id") and row.get("input_path")
    ]
    waiting = [row for row in tasks if row.get("status") in ACTIVE_STATUSES]
    if not waiting:
        return None

    task_directories = sorted({str(Path(row["input_path"]).parent) for row in waiting})
    batch_directories = sorted({str(Path(row["input_path"]).parent.parent) for row in waiting})
    upload_roots = {str(Path(path).parent) for path in batch_directories}
    stage_counts = {stage: sum(row.get("stage") == stage for row in waiting)
                    for stage in {row.get("stage") for row in waiting}}
    relax_only = set(stage_counts) == {"relax_and_feature"}
    return {
        "task_count": len(waiting),
        "waiting_task_count": len(waiting),
        "waiting_by_stage": stage_counts,
        "recovered_count": int(recovered_count or 0),
        "task_ids": [row.get("task_id") for row in waiting],
        "task_directories": task_directories,
        "batch_directories": batch_directories,
        "results_directories": sorted({str(Path(path).parent / "results")
                                        for path in batch_directories}),
        "upload_root": next(iter(upload_roots)) if len(upload_roots) == 1 else None,
        "upload_plan_path": (str(Path(next(iter(upload_roots))) / "RELAX_UPLOAD_PLAN.json")
                             if relax_only and len(upload_roots) == 1 and
                             (Path(next(iter(upload_roots))) / "RELAX_UPLOAD_PLAN.json").is_file()
                             else None),
    }
