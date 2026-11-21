"""Small evidence fixes: exact attribution, one attempted diagnostic, shared acceptance IDs."""

from copy import deepcopy
from unittest.mock import patch
import pytest

from tests.benchmark_small_evidence import Rows, budget_state, stages_state
from tests.test_phase_quality import sample
from phase_agent.evidence.round_budget_evidence import budget_decision_outcomes
from phase_agent.evidence.summarize_task_rounds import summarize_task_rounds
from phase_agent.evidence.summarize_task_stages import summarize_task_stages
from phase_agent.evidence.build_decision_context import build_decision_context
from phase_agent.analysis.phase import phase_quality_reports as quality


def test_budget_scans_source_collections_once():
    state = budget_state(12, 80, 16)
    budget_decision_outcomes(state)
    assert state["tasks"].passes == state["rewards"].passes == 1


def test_budget_indexes_preserve_duplicate_rows_order_versions_and_unknown_cost():
    state = {
        "action_records": [
            {
                "record_id": "d",
                "final_action": {"round_budget_review": {"choice": "new_search"}},
                "execution_result": {
                    "result": {
                        "tasks": [{"task_id": "b"}, {"task_id": "b"}, {"task_id": "missing"}]
                    }
                },
            }
        ],
        "tasks": [
            {
                "task_id": "b",
                "parent_decision_id": "d",
                "status": "completed",
                "model_version": "m1",
                "actual_cost": 0.1,
            },
            {
                "task_id": "a",
                "parent_decision_id": "d",
                "status": "failed",
                "model_version": "m2",
                "actual_cost": 0.2,
            },
            {"task_id": "b", "status": "completed", "model_version": "m1", "actual_cost": 0.3},
            {"task_id": "unrelated", "status": "completed", "actual_cost": 999},
            {"parent_decision_id": "d", "status": "completed", "actual_cost": 999},
        ],
        "rewards": [
            {"batch_id": "second", "task_ids": ["b", "a", "b"], "reward": 2},
            {"batch_id": "first", "task_ids": ["a"], "reward": 1},
            {"batch_id": "unrelated", "task_ids": ["missing"], "reward": 9},
        ],
        "processed_task_ids": [],
    }
    before = deepcopy(state)
    result = budget_decision_outcomes(state)[0]
    assert result["task_count"] == 3 and result["task_ids"] == ["a", "b"]
    assert result["actual_relative_cost"] == sum([0.1, 0.2, 0.3])
    assert result["model_version"] is None and result["status"] == "complete"
    assert [r["batch_id"] for r in result["observed_rewards"]] == ["second", "first"]
    assert state == before
    state["tasks"][0]["actual_cost"] = {"value": 1, "unit": "GPU-hour"}
    assert budget_decision_outcomes(state)[0]["actual_relative_cost"] is None
    state["tasks"][0].update(actual_cost=0, status="running")
    assert budget_decision_outcomes(state)[0]["status"] == "partial"
    state["tasks"][0]["status"] = "completed"
    assert summarize_task_stages(state)["stages"][0]["completed_unaccepted_count"] > 0


def test_recent_limit_matches_full_tail_without_dropping_task_or_reward_evidence():
    state = budget_state(20, 90, 30)
    state["action_records"].append({"record_id": "ignored", "final_action": {}})
    expected = budget_decision_outcomes(state)[-8:]
    state["tasks"].reset()
    state["rewards"].reset()
    assert budget_decision_outcomes(state, limit=8) == expected
    assert state["tasks"].passes == state["rewards"].passes == 1
    assert len(budget_decision_outcomes(state)) == 20


def test_no_review_does_not_scan_tasks_or_rewards():
    state = budget_state(2, 20, 4)
    state["action_records"] = []
    assert budget_decision_outcomes(state) == []
    assert state["tasks"].passes == state["rewards"].passes == 0


def test_round_summary_prepares_acceptance_once_and_keeps_order_and_acceptance():
    state = stages_state(20)
    before = deepcopy(state)
    expected = []
    from phase_agent.evidence.summarize_task_rounds import LINEAGE_FIELDS
    import json

    groups = {}
    for task in state["tasks"]:
        key = json.dumps({k: task.get(k) for k in LINEAGE_FIELDS}, sort_keys=True)
        groups.setdefault(key, []).append(task)
    for key in sorted(groups):
        expected.append(
            summarize_task_stages(
                {"tasks": groups[key], "processed_task_ids": list(before["processed_task_ids"])}
            )["stages"]
        )
    state["processed_task_ids"].reset()
    result = summarize_task_rounds(state)
    assert state["processed_task_ids"].passes == 1
    assert [g["stage_evidence"] for g in result["groups"]] == expected
    assert result == summarize_task_rounds({**state, "tasks": state["tasks"][::-1]})
    assert state == before


def test_standalone_stage_summary_remains_compatible():
    state = {
        "tasks": [
            {"task_id": "a", "stage": "dft_single_point", "status": "completed"},
            {"task_id": "b", "stage": "dft_single_point", "status": "completed"},
        ],
        "processed_task_ids": ["a"],
    }
    result = summarize_task_stages(state)
    row = result["stages"][0]
    assert row["accepted_count"] == row["completed_unaccepted_count"] == 1
    assert not row["all_completed_and_accepted"]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("synthetic malformed"),
        FileNotFoundError("synthetic missing"),
        OverflowError("synthetic overflow"),
    ],
)
def test_failed_quality_report_is_not_retried_within_context(error):
    state = sample()
    state["qbc_candidates"] = [
        {"candidate_id": "c", "model_version": "m1", "ehull": 0.01, "qbc": {"status": "completed"}}
    ]
    before = deepcopy(state)
    with patch.object(quality, "get_phase_quality_report", side_effect=error) as get:
        result = build_decision_context(state)
    assert get.call_count == 1
    diagnostic = result["current_phase_diagram"]["mlip"]["quality_diagnostics"]
    assert diagnostic["status"] == "diagnostic_unavailable" and diagnostic["reason"] == str(error)
    candidate = result["qbc_candidates"][0]
    assert candidate["phase_quality_signal"]["status"] == "unavailable"
    assert candidate["dft_sampling_evidence"]["status"] == "unavailable"
    assert candidate["ehull"] == 0.01 and candidate["qbc"]["status"] == "completed"
    assert state == before


def test_successful_quality_report_shared_and_cache_hit_still_fingerprints():
    state = sample()
    with patch.object(quality, "_REPORT_CACHE", quality.OrderedDict()):
        with patch.object(
            quality, "get_phase_quality_report", wraps=quality.get_phase_quality_report
        ) as get:
            first = build_decision_context(state)
            assert get.call_count == 1
        with (
            patch.object(
                quality, "diagnose_phase_diagram", side_effect=AssertionError("cache miss")
            ),
            patch.object(quality, "_digest", wraps=quality._digest) as fingerprint,
        ):
            second = build_decision_context(state)
            assert fingerprint.call_count == 1
        assert first == second


def test_explicit_none_keeps_legacy_fetch_behavior():
    state = sample()
    with patch.object(
        quality, "get_phase_quality_report", side_effect=ValueError("missing")
    ) as get:
        assert quality.quality_summary(state, report=None)["status"] == "diagnostic_unavailable"
        rows = quality.candidate_quality_evidence(
            state, [{"candidate_id": "c", "model_version": "m1"}], report=None
        )
        assert rows[0]["phase_quality_signal"]["status"] == "unavailable"
    assert get.call_count == 2
