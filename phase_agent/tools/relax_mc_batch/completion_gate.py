"""One formal-completion contract for imported Relax/MC batches and publications."""

import json
import zipfile


def completion_error(summary, tasks):
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        return "batch_status_not_completed"
    if not tasks or summary.get("task_count") != len(tasks):
        return "batch_task_scope_incomplete"
    if len({t.get("task_id") for t in tasks}) != len(tasks):
        return "batch_task_ids_not_unique"
    if any(t.get("status") != "completed" for t in tasks):
        return "batch_tasks_not_all_completed"
    return None


def require_complete_archive(archive):
    # This gate runs before extraction and before the already-imported shortcut.
    # The existing importer still verifies the archive and every signed result.
    with zipfile.ZipFile(archive) as zipped:
        summary = json.loads(zipped.read("result_summary.json"))
        returned = json.loads(zipped.read("workflow_state.json"))
    error = completion_error(summary, returned.get("tasks") or [])
    if error:
        raise ValueError("relax_mc_formal_recovery_blocked:" + error)


def incomplete_imports(state):
    model = state.get("active_model_version")
    result = []
    for key, batch in (state.get("relax_mc_batches") or {}).items():
        if not batch.get("import_checksum") or (model and batch.get("model_version") != model):
            continue
        tasks = [t for t in state.get("tasks", []) if t.get("relax_mc_bundle_id") == key]
        if batch.get("status") != "completed" or completion_error(
            batch.get("result_summary"), tasks
        ):
            result.append(key)
    return result


def require_complete_imports(state):
    blocked = incomplete_imports(state)
    if blocked:
        raise ValueError("relax_mc_formal_publication_blocked:" + ",".join(blocked))
