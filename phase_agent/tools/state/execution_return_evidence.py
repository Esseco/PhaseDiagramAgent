"""Bounded returned facts and paths; no full state, structures or replay payload."""


def return_evidence(execution, *, action=None, invocation_id=None):
    result = execution.get("result") or {}
    evidence = {"status": execution.get("status"), "error": execution.get("error")}
    if not isinstance(result, dict):
        return evidence
    evidence["result_status"] = result.get("status")
    evidence["task_key"] = result.get("task_key")
    returned_state = result.get("state") or {}
    registered = returned_state.get("tasks") or [] if isinstance(returned_state, dict) else []
    if isinstance(registered, dict):
        registered = list(registered.values())
    keys = {key for key in (result.get("task_key"), (action or {}).get("task_key")) if key}
    related = [
        row
        for row in registered
        if isinstance(row, dict)
        and (
            row.get("task_key") in keys
            or (
                invocation_id
                and invocation_id
                in {
                    row.get("invocation_id"),
                    row.get("approval_record_id"),
                    row.get("parent_decision_id"),
                }
            )
        )
    ]
    fields = {
        "task_id",
        "task_key",
        "status",
        "stage",
        "input_path",
        "result_path",
        "directory",
        "model_version",
        "config_version",
        "invocation_id",
        "parent_decision_id",
        "approval_record_id",
    }
    evidence["returned_task_refs"] = [
        {key: value for key, value in row.items() if key in fields} for row in related
    ]
    evidence["task_ids"] = [
        row.get("task_id") for row in result.get("tasks") or [] if isinstance(row, dict)
    ]
    paths = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "state":
                    continue
                if isinstance(item, str) and (key.endswith("_path") or key == "directory"):
                    paths.add(item)
                elif isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    visit(item)

    visit(result)
    visit(evidence["returned_task_refs"])
    evidence["artifacts"] = [{"artifact_path": path} for path in sorted(paths)]
    return evidence
