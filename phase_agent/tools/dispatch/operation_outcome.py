"""Compact facts returned by a node, without inventing scientific completion."""


def operation_outcome(action, execution, context):
    result = execution.get("result")
    result = result if isinstance(result, dict) else {}
    state = context.get("event_state") or {}
    config = context.get("effective_config") or {}
    model = state.get("active_model") or config.get("mlip") or {}
    rows = [result]
    for key in ("batches", "handoffs"):
        values = result.get(key) or []
        if isinstance(values, list):
            rows.extend(value for value in values if isinstance(value, dict))
    ids, artifacts = [], []
    path_keys = (
        "folder",
        "directory",
        "upload_directory",
        "upload_archive",
        "results_directory",
        "return_directory",
        "import_directory",
        "manifest_path",
    )
    for row in rows:
        raw_ids = row.get("task_ids") or []
        if isinstance(raw_ids, list):
            ids.extend(value for value in raw_ids if isinstance(value, str))
        tasks = row.get("tasks") or []
        if isinstance(tasks, list):
            ids.extend(
                t["task_id"]
                for t in tasks
                if isinstance(t, dict) and isinstance(t.get("task_id"), str)
            )
        for key in path_keys:
            if isinstance(row.get(key), str):
                artifact = {"kind": key, "path": row[key]}
                if artifact not in artifacts:
                    artifacts.append(artifact)
    returned_state = result.get("state") or {}
    returned_model = returned_state.get("active_model") or {}
    return {
        "schema_version": 1,
        "operation": action.get("operation"),
        "tool": action.get("tool"),
        "task_key": action.get("task_key"),
        "config_version": context.get("config_version") or state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version")
        or model.get("version")
        or model.get("name"),
        "result_model_version": returned_state.get("active_model_version")
        or returned_model.get("version"),
        "dispatch_status": execution.get("status"),
        "result_status": result.get("status"),
        "task_ids": list(dict.fromkeys(ids)),
        "artifacts": artifacts,
        "submitted": result.get("submitted"),
        "error": execution.get("error"),
    }
