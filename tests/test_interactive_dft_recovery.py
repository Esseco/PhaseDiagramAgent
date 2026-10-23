"""Regression guards for real returned pairs and dialogue observation wiring."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phase_agent.tools.remote.batch_runner import RemoteBatchRunner, _task_result_path
from phase_agent.tools.workflows.refresh_dft_comparisons import refresh_dft_comparisons
from phase_agent.tools.workflows.create_dft_comparison_evaluator import (
    create_dft_comparison_evaluator,
)
from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment


def co_state():
    return json.loads(
        Path("E:/0-PhaseDiagram/O3-1ele-NaCoO2/workflow_state/state.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.parametrize("value", [".", "..", "", None])
def test_invalid_result_pointer_is_missing_not_a_directory_crash(tmp_path, value):
    state = {
        "tasks": [
            {
                "task_id": "D1",
                "task_key": "key",
                "stage": "dft_single_point",
                "status": "failed",
                "result_path": value,
            }
        ]
    }
    result = RemoteBatchRunner(tmp_path, worker_command=[]).collect_results_with_report(state)
    assert not result["results"]
    assert result["report"]["missing_result_count"] == 1
    assert _task_result_path(state["tasks"][0], None) is None


@pytest.mark.real_project
def test_refresh_saved_pairs_retains_task_identity_and_labels():
    state = co_state()
    training = deepcopy(state["dft_training_records"])
    pending = deepcopy(state["new_dft_records"])
    assessment = post_dft_assessment(state, state["confirmed_config"])
    result = refresh_dft_comparisons(
        state,
        assessment=assessment,
        evaluator=create_dft_comparison_evaluator(state["confirmed_config"], state=state),
    )
    assert post_dft_assessment(result, state["confirmed_config"])["status"] == "evaluated"
    assert result["dft_training_records"] == training and result["new_dft_records"] == pending


@pytest.mark.real_project
def test_refresh_explicit_old_scope_recovers_all_saved_groups():
    state = co_state()
    ids = [r["task_id"] for r in state["dft_dataset_records"]]
    evaluator = create_dft_comparison_evaluator(state["confirmed_config"], state=state)
    result = refresh_dft_comparisons(state, assessment=None, evaluator=evaluator, task_ids=ids)
    assert sum(r["status"] == "completed" for r in result["dft_mlip_comparisons"]) == 30
    repeated = refresh_dft_comparisons(result, assessment=None, evaluator=evaluator, task_ids=ids)
    assert repeated["dft_training_records"] == result["dft_training_records"]
    assert repeated["final_frame_dft_errors"] == result["final_frame_dft_errors"]
