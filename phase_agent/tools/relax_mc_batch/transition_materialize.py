"""Build immutable Relax evidence for a threshold-transition import."""

from pathlib import Path

from phase_agent.tools.relax_mc_batch.files import read, rebase
from phase_agent.tools.relax_mc_batch.transition_effective import effective_relax_result
from phase_agent.tools.remote.integrity import verify_result, file_checksum
from phase_agent.tools.remote.resolve_local_relax_structure import resolve_local_relax_structure
from phase_agent.tools.remote.transfer_bundle import checksum, import_bundle
from phase_agent.tools.state.normalize_task_result import normalize_task_result


def materialize_verified_relax(archive, expected_archive_sha256, directory, review):
    """Return effective result views and leave signed original bytes intact.

    The caller must independently validate the registered transition and its
    remote-input attestation. This helper does not change the project state,
    ledger, budget, approval or submission status.
    """
    if checksum(archive) != expected_archive_sha256:
        raise ValueError("transition_source_archive_changed")
    directory = Path(directory).resolve()
    import_bundle(archive, directory, extra_result_suffixes=(".csv", ".bin"))
    source = rebase(read(directory / "workflow_state.json"), directory)
    tasks = source.get("tasks") or []
    rows = review.get("rows") or []
    by_task = {row.get("task_id"): row for row in rows}
    if (
        not tasks
        or len(by_task) != len(rows)
        or len(tasks) != len(by_task)
        or {t.get("task_id") for t in tasks} != set(by_task)
    ):
        raise ValueError("transition_materialization_scope_mismatch")
    results = []
    for task in tasks:
        row = by_task[task["task_id"]]
        path = Path(task["result_path"]).resolve()
        if not path.is_relative_to(directory) or path.name != "result.json":
            raise ValueError("transition_result_path_invalid")
        verified = verify_result(path, path.with_name("task.finished.json"), task)
        if not verified["valid"] or file_checksum(path) != row["result_sha256"]:
            raise ValueError("transition_result_changed:" + task["task_id"])
        final = path.with_name("initial_relaxed.vasp")
        if not final.is_file() or file_checksum(final) != row["final_sha256"]:
            raise ValueError("transition_final_structure_changed:" + task["task_id"])
        view = effective_relax_result(verified["result"], row)
        view = resolve_local_relax_structure(view, path.parent)
        if view is None:
            raise ValueError("transition_final_structure_unavailable:" + task["task_id"])
        results.append(
            normalize_task_result(
                {**task, "object_id": task.get("object_id", task["structure_id"])}, view
            )
        )
    return {
        "status": "verified_relax_views",
        "results": results,
        "source_archive_sha256": expected_archive_sha256,
        "submitted": False,
    }
