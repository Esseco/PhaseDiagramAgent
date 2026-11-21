import pytest

from phase_agent.evidence.relaxation_extrapolation_round import assess_round


def rows(n, failures):
    return [{"structure_id": str(i), "monitor_complete": True,
             "extrapolation_failure": i < failures} for i in range(n)]


def test_threshold_and_distinct_structures():
    data = rows(20, 2)
    result = assess_round(data + [data[0]], scope_complete=True)
    assert result["structure_count"] == 20
    assert result["failure_fraction"] == .1
    assert result["direct_sampling_required"] is True
    assert assess_round(rows(21, 2), scope_complete=True)["direct_sampling_required"] is False


def test_missing_scope_or_conflicting_monitor_is_unknown():
    data = rows(10, 1)
    assert assess_round(data, scope_complete=False)["failure_fraction"] is None
    assert assess_round(data + [{**data[0], "extrapolation_failure": False}], scope_complete=True)["direct_sampling_required"] is None
    data[1]["monitor_complete"] = False
    assert assess_round(data, scope_complete=True)["status"] == "unknown"


def test_bad_identity_and_empty_scope():
    with pytest.raises(ValueError, match="missing_structure_id"):
        assess_round([{}], scope_complete=True)
    assert assess_round([], scope_complete=True)["direct_sampling_required"] is None
