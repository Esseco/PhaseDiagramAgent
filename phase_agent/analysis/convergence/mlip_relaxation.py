"""Physical Relax acceptance, separate from process/MC segment completion."""

from copy import deepcopy
from math import isfinite

DISCARDED = "discarded_unconverged"
EXCLUDED = "excluded_unconverged_relaxation"


def relaxation_failure(result, *, require_optimizer=False):
    if result.get("status") not in (None, "completed"):
        return None
    if result.get("stage") != "relax_and_feature" or result.get("energy_method") == "dft":
        return None
    output = result.get("outputs") or result
    operation = (result.get("worker_job") or {}).get("operation") or (
        result.get("parameters") or {}
    ).get("model_refresh_operation")
    if operation == "predict" or (
        output.get("single_point_completed") is True and "optimizer_converged" not in output
    ):
        return None
    physical = output.get("optimizer_converged")
    if physical is False:
        return "optimizer_not_converged"
    if require_optimizer and physical is not True:
        return "optimizer_convergence_not_verified"
    if physical is not None and physical is not True:
        return "invalid_optimizer_convergence_evidence"
    if physical is True:
        # Cell filters optimize transformed atom/stress forces; raw atomic fmax
        # alone must not overturn the optimizer's explicit convergence verdict.
        return None
    force = output.get("force_max_ev_per_angstrom")
    target = output.get("fmax_target_ev_per_angstrom")
    if target is None:
        target = (result.get("parameters") or {}).get("fmax")
    if force is not None and target is not None:
        try:
            if not isfinite(float(force)) or not isfinite(float(target)) or float(target) <= 0:
                return "invalid_relaxation_force_evidence"
            if float(force) > float(target) + 1e-12:
                return "relaxation_force_above_target"
        except (TypeError, ValueError, OverflowError):
            return "invalid_relaxation_force_evidence"
    return None


def reject_relaxation(result, reason):
    result.update(converged=False, checks_passed=False, result_disposition=DISCARDED)
    result["relaxation_quality"] = {"status": "rejected", "reason": reason}


def reconcile_relaxation_quality(state, *, _owned=False):
    """Correct proven contradictions; retain files, raw outputs, costs and history."""
    current = state if _owned else deepcopy(state)
    rejected = {}
    for task in current.get("tasks") or []:
        if task.get("task_id") in (current.get("scientific_feedback_errors") or {}):
            continue
        reason = relaxation_failure(task)
        if reason:
            reject_relaxation(task, reason)
            if task.get("task_id"):
                rejected[task["task_id"]] = task
    records = []
    for record in current.get("phase_records") or []:
        task = rejected.get(record.get("source_task_id"))
        if task is not None:
            record.update(status=EXCLUDED, converged=False, checks_passed=False)
            record["relaxation_quality"] = deepcopy(task["relaxation_quality"])
            records.append(record.get("record_id"))
    if not rejected:
        return current
    bad_records = set(records)
    diagram = (current.get("phase_diagrams") or {}).get("mlip") or {}
    for snapshot in [diagram, *(current.get("phase_diagrams_by_model") or {}).values()]:
        if bad_records.intersection(e.get("record_id") for e in snapshot.get("entries") or []):
            snapshot.update(status="quality_review_required", reason=EXCLUDED)
            if snapshot is diagram:
                current["qbc_candidates"] = []
    bad_paths = {(t.get("outputs") or {}).get("structure_path") for t in rejected.values()} - {None}
    pools = current.get("branch_hull_batches") or {}
    invalid = {
        version
        for version, pool in pools.items()
        if any(
            r.get("source_task_id") in rejected or r.get("structure_path") in bad_paths
            for r in pool.get("records") or []
        )
    }
    for model, version in list((current.get("current_branch_hull_by_model") or {}).items()):
        if version in invalid:
            current["current_branch_hull_by_model"].pop(model)
    if current.get("current_branch_hull_version") in invalid:
        current["current_branch_hull_version"] = None
    current["mlip_relaxation_quality"] = {
        "discarded_task_ids": sorted(rejected),
        "excluded_record_ids": sorted(r for r in records if r),
    }
    return current
