from copy import deepcopy


def effective_relax_result(raw_result, review):
    """Derive a provenance-bound transient view; leave the signed result unchanged."""
    from phase_agent.tools.relax_mc_batch.threshold_transition import reassess_verified_task

    derived_review = reassess_verified_task(raw_result, raw_result)
    for key in (
        "task_id",
        "structure_id",
        "model_version",
        "config_version",
        "original_failure",
        "effective_failure",
        "final_feature_difference",
        "effective_failure_events",
    ):
        if derived_review.get(key) != review.get(key):
            raise ValueError("transition_effective_review_mismatch:" + key)
    result = deepcopy(raw_result)
    outputs = result["outputs"]
    monitor = outputs["extrapolation_monitor"]
    outputs["extrapolation_monitor"] = {
        **monitor,
        "feature_difference_threshold": 3.0,
        "extrapolation_failure": review["effective_failure"],
        "failures": deepcopy(review["effective_failure_events"]),
    }
    outputs["relaxation_observation"] = {
        **outputs["relaxation_observation"],
        "extrapolation_failure": review["effective_failure"],
    }
    if review["original_failure"] and not review["effective_failure"]:
        if (
            result.get("result_disposition") != "excluded_extrapolation_failure"
            or result.get("checks_passed") is not False
            or result.get("relaxation_quality")
            != {"status": "rejected", "reason": "extrapolation_failure"}
        ):
            raise ValueError("transition_other_rejection_cannot_be_cleared")
        result.pop("result_disposition")
        result.pop("checks_passed")
        result.pop("relaxation_quality")
    result["threshold_transition"] = {
        "old_threshold": 1.0,
        "new_threshold": 3.0,
        "source_result_sha256": review["result_sha256"],
    }
    return result
