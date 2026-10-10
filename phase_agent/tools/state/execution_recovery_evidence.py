"""Read registered artifacts without inferring that an interrupted action completed."""

import hashlib
from pathlib import Path


def inspect_execution(state, issue, *, state_path=None):
    identity = issue.get("identity") or {}
    invocation = identity.get("invocation_id")
    records = [
        row
        for row in state.get("action_records") or []
        if invocation and row.get("record_id") == invocation
    ]
    from phase_agent.tools.state.execution_receipts import execution_events

    journal = execution_events(state_path, invocation) if state_path and invocation else []
    actions = [
        row.get("final_action") or (row.get("agent_proposal") or {}).get("raw_action") or {}
        for row in records
    ]
    actions.extend(
        row["payload"].get("action") or {} for row in journal if row["event"] == "claimed"
    )
    keys = {action.get("task_key") for action in actions if action.get("task_key")}
    if issue.get("task_key"):
        keys.add(issue["task_key"])
    returned_ids = {
        task_id
        for event in journal
        for task_id in (event["payload"].get("evidence") or {}).get("task_ids", [])
    }
    returned_ids.update(
        row.get("task_id")
        for event in journal
        for row in (event["payload"].get("evidence") or {}).get("returned_task_refs", [])
        if row.get("task_id")
    )
    tasks = state.get("tasks") or []
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    related = [
        row
        for row in tasks
        if (
            row.get("task_key") in keys
            or row.get("task_id") in returned_ids
            or (
                invocation
                and invocation
                in {
                    row.get("invocation_id"),
                    row.get("approval_record_id"),
                    row.get("parent_decision_id"),
                }
            )
        )
    ]
    unregistered = sorted(
        str(item)
        for item in returned_ids
        if item and item not in {row.get("task_id") for row in tasks}
    )
    files = {}

    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and (key.endswith("_path") or key == "directory"):
                    path = Path(item)
                    # Relative paths cannot safely be interpreted from the application's cwd.
                    if not path.is_absolute():
                        files[item] = {"path": item, "status": "relative_path_requires_owner"}
                        continue
                    try:
                        status = (
                            "directory"
                            if path.is_dir()
                            else "file"
                            if path.is_file()
                            else "missing"
                        )
                        row = {"path": str(path), "status": status}
                        if status == "file":
                            row["size"] = path.stat().st_size
                            row["mtime_ns"] = path.stat().st_mtime_ns
                            if row["size"] <= 8 * 1024 * 1024:
                                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        files[str(path)] = row
                    except OSError as error:
                        files[item] = {
                            "path": item,
                            "status": "unreadable",
                            "error": type(error).__name__,
                        }
                elif isinstance(item, (dict, list)):
                    paths(item)
        elif isinstance(value, list):
            for item in value:
                paths(item)

    for row in journal:
        paths(row["payload"].get("evidence") or {})
    for task in related:
        paths(task)
    for record in records:
        paths(record.get("execution_result") or record.get("execution") or {})
    completed = [row.get("task_id") for row in related if row.get("status") == "completed"]
    pending = [
        row.get("task_id")
        for row in related
        if row.get("status") in {"pending", "running", "submitted"}
    ]
    task_ids = {row.get("task_id") for row in related if row.get("task_id")}
    phase_records = [
        row.get("record_id")
        for row in state.get("phase_records") or []
        if row.get("source_task_id") in task_ids
    ]
    reservations = [
        {"task_key": key, **state.get("budget_reservations", {}).get(key, {})}
        for key in sorted(keys)
        if key in state.get("budget_reservations", {})
    ]
    next_action = (
        "核对已登记结果并复用，修复缺失入账后再推进"
        if completed
        else "核对已有任务是否提交，回传结果；不要重复准备或提交"
        if pending
        else "补充原动作的输出目录与任务来源，核对文件和外部作业后决定恢复"
    )
    if unregistered:
        next_action = "工具曾返回任务引用，但当前登记缺失；先核对任务来源并补全登记，不能重复提交或声明无副作用"
    if any(
        row["status"] in {"missing", "unreadable", "relative_path_requires_owner"}
        for row in files.values()
    ):
        next_action = "先核对缺失或无法读取的文件及路径迁移，再决定复用或修复台账"
    return {
        "identity": identity,
        "execution_events": journal,
        "task_keys": sorted(keys),
        "reason": issue.get("reason"),
        "phase": issue.get("phase"),
        "completed_task_ids": completed,
        "pending_task_ids": pending,
        "task_count": len(related),
        "unregistered_returned_task_ids": unregistered,
        "artifact_checks": list(files.values()),
        "budget_reservations": reservations,
        "reserved_relative_cost": state.get("reserved_relative_cost", 0),
        "phase_record_ids": phase_records,
        "next_action": next_action,
        "automatic_replay_allowed": False,
        "limitation": "文件存在和任务完成记录不证明整个动作已完成；缺失文件不证明没有副作用。",
    }
