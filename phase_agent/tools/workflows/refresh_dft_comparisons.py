"""Retry missing same-frame predictions without recollecting or charging DFT."""

from copy import deepcopy
from phase_agent.analysis.feedback.dft_result_products import record_dft_products


def refresh_dft_comparisons(state, *, assessment, evaluator, manager=None, task_ids=None):
    if not callable(evaluator):
        return state
    if task_ids is None:
        if not assessment or assessment["status"] == "evaluated":
            return state
        task_ids = assessment["recovered_task_ids"]
    current = deepcopy(state)
    training = deepcopy(current.get("dft_training_records") or [])
    pending = deepcopy(current.get("new_dft_records") or [])
    ids = set(task_ids)
    tasks = {row.get("task_id"): row for row in current.get("tasks") or []}
    completed = {
        row.get("task_id")
        for row in current.get("dft_mlip_comparisons") or []
        if row.get("status") == "completed"
    }
    for record in list(current.get("dft_dataset_records") or []):
        if (
            record.get("task_id") not in ids
            or record.get("task_id") in completed
            or record.get("checks_passed") is not True
            or record.get("training_ready") is not True
            or record.get("status") != "completed"
            or record.get("converged") is not True
        ):
            continue
        # Legacy dataset records omitted task_key. Recover identity from the
        # bound task, never from the prediction's self-reported identity.
        task = tasks.get(record.get("task_id")) or {}
        result = {**deepcopy(task), **deepcopy(record), "outputs": deepcopy(record)}
        if not result.get("task_key"):
            result["task_key"] = task.get("task_key")
        record_dft_products(current, result, evaluator=evaluator, manager=manager, refresh=True)
    # Prediction retries are not new labels, eligibility decisions or costs.
    current["dft_training_records"] = training
    current["new_dft_records"] = pending
    return current
