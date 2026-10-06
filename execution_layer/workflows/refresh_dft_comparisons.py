"""Retry missing same-frame predictions without recollecting or charging DFT."""
from copy import deepcopy
from analysis_layer.feedback.dft_result_products import record_dft_products


def refresh_dft_comparisons(state, *, assessment, evaluator, manager=None):
    if not assessment or assessment["status"] == "evaluated" or not callable(evaluator):
        return state
    current = deepcopy(state)
    training = deepcopy(current.get("dft_training_records") or [])
    pending = deepcopy(current.get("new_dft_records") or [])
    ids = set(assessment["recovered_task_ids"])
    completed = {row.get("task_id") for row in current.get("dft_mlip_comparisons") or []
                 if row.get("status") == "completed"}
    for record in list(current.get("dft_dataset_records") or []):
        if (record.get("task_id") not in ids or record.get("task_id") in completed
                or record.get("checks_passed") is not True or record.get("training_ready") is not True
                or record.get("status") != "completed" or record.get("converged") is not True):
            continue
        result = {**deepcopy(record), "outputs": deepcopy(record)}
        record_dft_products(current, result, evaluator=evaluator, manager=manager, refresh=True)
    # Prediction retries are not new labels, eligibility decisions or costs.
    current["dft_training_records"] = training
    current["new_dft_records"] = pending
    return current
