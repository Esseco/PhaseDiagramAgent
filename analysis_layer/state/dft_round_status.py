"""Factual per-operation DFT recovery counts; missing returns are not failures."""
import json

DFT_STAGES = {"dft_single_point", "dft_relax"}
SCOPE_FIELDS = ("model_version", "search_group_index", "parent_relax_round", "upload_operation_id")


def dft_round_scope(row):
    scope = {key: row.get(key) for key in SCOPE_FIELDS}
    if not scope["upload_operation_id"]:
        scope["batch_id"] = row.get("batch_id") or row.get("task_id")
    return scope


def dft_recovery_rounds(state):
    groups = {}
    received = set(state.get("processed_task_ids") or []) | set(state.get("feedback_processed_task_ids") or [])
    received.update(row.get("task_id") for row in state.get("dft_dataset_records") or [])
    for task in state.get("tasks") or []:
        if task.get("stage") in DFT_STAGES and task.get("task_id"):
            scope = dft_round_scope(task)
            key = json.dumps(scope, sort_keys=True)
            group = groups.setdefault(key, {"scope": scope, "tasks": {}})
            group["tasks"][task["task_id"]] = task
    output = []
    for key, group in sorted(groups.items()):
        tasks = list(group["tasks"].values())
        # Legacy terminal task records are explicit evidence too, but a locally
        # cancelled task is not a returned DFT calculation.
        returned = [t for t in tasks if t["task_id"] in received
                    or t.get("status") in {"completed", "failed", "timeout"}]
        pending = [t for t in tasks if t.get("status") in {"pending", "running", "submitted", "unknown"}
                   and t.get("recovery_wait_waived") is not True]
        waived = [t for t in tasks if t.get("recovery_wait_waived") is True and t not in returned]
        output.append({"scope_key": key, "scope": group["scope"], "expected_tasks": len(tasks),
                       "recovered_tasks": len(returned), "recovery_ratio": len(returned)/len(tasks),
                       "successful_tasks": sum(t.get("status") == "completed" for t in returned),
                       "failed_tasks": sum(t.get("status") in {"failed", "timeout"} for t in returned),
                       "pending_task_ids": sorted(t["task_id"] for t in pending),
                       "waived_task_ids": sorted(t["task_id"] for t in waived),
                       "recovered_task_ids": sorted(t["task_id"] for t in returned)})
    return output
