import copy
import pytest
from phase_agent.tools.relax_mc_batch.threshold_transition import reassess_verified_task
from phase_agent.tools.relax_mc_batch.transition_effective import effective_relax_result


def payload():
    t = {
        "task_id": "t",
        "task_key": "k",
        "stage": "relax_and_feature",
        "structure_id": "s",
        "model_version": "m",
        "config_version": "c",
        "status": "completed",
        "converged": True,
        "checks_passed": False,
        "result_disposition": "excluded_extrapolation_failure",
        "relaxation_quality": {"status": "rejected", "reason": "extrapolation_failure"},
        "outputs": {
            "optimizer_converged": True,
            "extrapolation_monitor": {
                "extrapolation_failure": True,
                "final_feature_difference": 2.0,
                "failures": [{"trigger": "final", "score": 2.0}],
            },
            "relaxation_observation": {
                "identity_scheme": "periodic_input_v1",
                "monitor_complete": True,
                "structure_id": "hash",
                "extrapolation_failure": True,
            },
        },
    }
    review = {**reassess_verified_task(t, t), "result_sha256": "a" * 64}
    return t, review


def test_derived_view_preserves_signed_source_and_removes_only_obsolete_rejection():
    raw, review = payload()
    before = copy.deepcopy(raw)
    result = effective_relax_result(raw, review)
    assert (
        raw == before
        and result["outputs"]["extrapolation_monitor"]["extrapolation_failure"] is False
    )
    assert result["outputs"]["extrapolation_monitor"]["feature_difference_threshold"] == 3
    assert "checks_passed" not in result and "result_disposition" not in result
    assert result["threshold_transition"]["source_result_sha256"] == "a" * 64


def test_unrelated_rejection_and_review_mismatch_do_not_clear():
    raw, review = payload()
    raw["relaxation_quality"]["reason"] = "other"
    with pytest.raises(ValueError, match="other_rejection"):
        effective_relax_result(raw, review)
    raw, review = payload()
    review["final_feature_difference"] = 1.0
    with pytest.raises(ValueError, match="review_mismatch"):
        effective_relax_result(raw, review)
