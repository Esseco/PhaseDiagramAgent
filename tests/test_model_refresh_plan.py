import pytest

from phase_agent.analysis.feedback.model_refresh_plan import build_model_refresh_plan, supplemental_relaxation_required
from phase_agent.science.training.cumulative_training_records import cumulative_training_records


def row(index, energy, **extra):
    return {"structure_id": str(index), "structure_sha256": str(index), "ehull": energy,
            "source_version": "old", "phase": "O3", "x_Na_per_O2": .5, **extra}


def test_thresholds_and_duplicate_structure():
    result = build_model_refresh_plan([row(0, 0), row(1, .001), row(2, .010), row(2, .010),
                                      row(3, None)], old_version="old", new_version="new")
    assert [r["operation"] for r in result["candidates"]] == ["relax", "predict", "predict", "predict"]
    assert result["total_unique"] == 4
    assert result["maximum_supplemental_count"] == 3


def test_far_sampling_is_global_ten_percent_not_per_stratum():
    rows = [row(i, .02, phase=f"phase{i}") for i in range(21)]
    result = build_model_refresh_plan(rows, old_version="old", new_version="new")
    assert len(result["candidates"]) == 3
    assert len(result["coverage_gaps"]) == 18
    assert result == build_model_refresh_plan(reversed(rows), old_version="old", new_version="new")


def test_historical_near_is_kept_and_wrong_model_rejected():
    result = build_model_refresh_plan([row(0, .2, historical_near_hull=True)], old_version="old", new_version="new")
    assert result["candidates"][0]["selection_reason"] == "historical_near_hull"
    with pytest.raises(ValueError):
        build_model_refresh_plan([row(0, .1, source_version="other")], old_version="old", new_version="new")


def test_supplemental_force_is_atomic_norm():
    assert supplemental_relaxation_required(.0009, [])
    assert supplemental_relaxation_required(.001, [[.04, .04, 0]])
    assert not supplemental_relaxation_required(.01, [[.05, 0, 0]])
    assert not supplemental_relaxation_required(.011, [[100, 0, 0]])


def test_cumulative_records_only_merge_mirrored_identity():
    first = {"data_id": "a", "checks_passed": True}
    second = {"data_id": "b", "checks_passed": True}
    result = cumulative_training_records({"dft_training_records": [first, second], "new_dft_records": [second]})
    assert len(result) == 2
    assert [r["is_current_round"] for r in result] == [False, True]
    assert "is_current_round" not in second
