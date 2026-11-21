import copy
import pytest
from phase_agent.tools.relax_mc_batch.threshold_transition import (
    reassess_legacy_monitor,
    reassess_verified_task,
)


def monitor(final, events):
    return {
        "final_feature_difference": final,
        "failures": events,
        "extrapolation_failure": bool(events),
    }


def test_keeps_original_and_reassesses_all_events():
    original = monitor(
        2.5, [{"trigger": "force_jump", "score": 3.2}, {"trigger": "final", "score": 2.5}]
    )
    snapshot = copy.deepcopy(original)
    result = reassess_legacy_monitor(original)
    assert (
        original == snapshot
        and result["original_failure"] is True
        and result["effective_failure"] is True
    )
    assert result["effective_failure_events"] == [{"trigger": "force_jump", "score": 3.2}]


def test_exact_three_is_not_failure():
    m = monitor(3.0, [{"trigger": "final", "score": 3.0}])
    assert reassess_legacy_monitor(m)["effective_failure"] is False


@pytest.mark.parametrize(
    "m",
    [
        {"final_feature_difference": float("nan"), "extrapolation_failure": False, "failures": []},
        {"final_feature_difference": 2.0, "extrapolation_failure": False, "failures": []},
        {
            "final_feature_difference": 2.0,
            "extrapolation_failure": True,
            "failures": [{"trigger": "final", "score": 1.5}],
        },
        {
            "final_feature_difference": 0.5,
            "extrapolation_failure": False,
            "failures": [{"trigger": "force_jump", "score": 4.0}],
        },
        {
            "final_feature_difference": 0.5,
            "extrapolation_failure": True,
            "failures": [{"trigger": "unknown", "score": 4.0}],
        },
    ],
)
def test_invalid_monitor_evidence_is_rejected(m):
    with pytest.raises(ValueError):
        reassess_legacy_monitor(m)


def test_exact_task_identity_and_optimizer_are_required():
    keys = {
        "task_id": "t",
        "task_key": "k",
        "stage": "relax_and_feature",
        "structure_id": "s",
        "model_version": "m",
        "config_version": "c",
    }
    result = {
        **keys,
        "status": "completed",
        "converged": True,
        "outputs": {
            "optimizer_converged": True,
            "extrapolation_monitor": monitor(2, [{"trigger": "final", "score": 2}]),
            "relaxation_observation": {
                "identity_scheme": "periodic_input_v1",
                "monitor_complete": True,
                "structure_id": "hash",
                "extrapolation_failure": True,
            },
        },
    }
    assert reassess_verified_task(keys, result)["effective_failure"] is False
    for mutation in ({"task_id": "wrong"}, {"converged": False}, {"stage": "deep_search"}):
        with pytest.raises(ValueError):
            reassess_verified_task(keys, {**result, **mutation})
