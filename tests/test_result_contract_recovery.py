from copy import deepcopy
import json

import pytest

from tests.test_stability_boundaries import manager_and_result
from phase_agent.persistence.ledger.result_validation import (
    InvalidScientificResult,
    validate_scientific_result,
)
from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback


@pytest.mark.parametrize("summary", [[], [{"rank": 0, "energy_mean_per_atom": -2.5}]])
def test_worker_pool_summary_is_a_list_of_objects(summary):
    manager, result = manager_and_result()
    result["outputs"]["pool_summary"] = summary
    validate_scientific_result(result)
    answer = apply_scientific_feedback({}, [result], manager=manager)
    assert len(answer["recorded"]) == 1
    assert not answer["state"]["scientific_feedback_errors"]


@pytest.mark.parametrize("summary", [{}, "bad", [None], [3], [[{}]]])
def test_malformed_pool_summary_is_rejected(summary):
    _, result = manager_and_result()
    result["outputs"]["pool_summary"] = summary
    with pytest.raises(InvalidScientificResult, match="pool_summary"):
        validate_scientific_result(result)


@pytest.mark.parametrize(
    "fault", [None, "energy", "model_version", "checks_passed", "status", "structure_sha256"]
)
def test_excluded_saved_record_is_revalidated_without_duplicate_accounting(fault):
    manager, result = manager_and_result()
    result["outputs"].update(actual_phase="O3", phase_identification={"status": "identified"})
    first = apply_scientific_feedback({}, [result], manager=manager)
    state = first["state"]
    record = state["phase_records"][0]
    record.update(status="excluded_invalid_result", checks_passed=False)
    state["scientific_feedback_errors"] = {
        result["task_id"]: {"code": "result_output_object_required:pool_summary"}
    }
    if fault == "energy":
        record["energy"] -= 1
    elif fault == "model_version":
        record["model_version"] = "different"
    elif fault == "checks_passed":
        state["tasks"][0]["checks_passed"] = False
    elif fault == "status":
        state["tasks"][0]["status"] = "failed"
    elif fault == "structure_sha256":
        record["structure_sha256"] = "different-geometry"
    before = deepcopy(manager.data)
    costs = deepcopy(state.get("cost_history"))
    second = apply_scientific_feedback(state, [], manager=manager)
    third = apply_scientific_feedback(second["state"], [], manager=manager)
    restored = third["state"]["phase_records"][0]
    assert len(third["state"]["phase_records"]) == 1
    assert manager.data == before
    assert third["state"].get("cost_history") == costs
    assert not second["recorded"] and not third["recorded"]
    assert (restored["status"] != "excluded_invalid_result") is (fault is None)
    if fault is None:
        assert restored["status"] == "pending_phase_identification"
    assert restored["checks_passed"] is (fault is None)


def test_history_refresh_warning_is_separate_and_available_in_diagnostics(tmp_path):
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    from phase_agent.runtime.project_diagnostics import inspect_project

    collection = {
        "invalid_count": 0,
        "metadata_refresh_report": {
            "invalid_count": 15,
            "details_preview": [{"task_id": "old-dft", "reason": "binding_mismatch"}],
        },
    }
    state = {"tasks": [], "result_collection": collection}
    reply = format_workflow_reply(
        {
            "status": "awaiting_manual_submission",
            "state": state,
            "manual_wait": {
                "waiting_by_stage": {"deep_search": 5},
                "result_collection": collection,
            },
        },
        tmp_path / "state.json",
    )
    assert "历史 DFT 元数据补充校验未通过 15 项" in reply
    assert "校验/结构问题 15" not in reply
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    facts = inspect_project(tmp_path / "state.json")["sections"]["state"]["data"]
    assert facts["result_collection"] == collection


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("during_error_reply", [False, True])
def test_client_abort_does_not_raise_or_reclassify_as_business_failure(method, during_error_reply):
    from phase_agent.runtime.agent_api import create_server

    server = create_server(lambda *a, **k: None, api_key="a" * 24, port=0)
    handler = object.__new__(server.RequestHandlerClass)
    handler.path = "/test"
    seen = []

    def fail():
        raise ValueError("business failure") if during_error_reply else ConnectionAbortedError()

    def boundary(error, **kwargs):
        seen.append(error)
        raise ConnectionAbortedError()

    handler._get = handler._post = fail
    if during_error_reply:
        handler._json = lambda *a: boundary(None)
    else:
        handler._boundary_failure = boundary
    try:
        getattr(handler, "do_" + method)()
        assert bool(seen) is during_error_reply
    finally:
        server.server_close()


def test_rejected_old_dft_refresh_does_not_count_as_missing_current_result(tmp_path):
    from tests.test_dft_spin_acceptance import make_result
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.remote.integrity import file_checksum

    incoming = make_result()
    taskdir = tmp_path / "old" / "00000-D1"
    taskdir.mkdir(parents=True)
    (taskdir / "task.json").write_text(
        json.dumps({"task_id": "D1", "task_key": "K1", "model_version": "m1"})
    )
    path = taskdir / "result.json"
    path.write_text(json.dumps(incoming))
    (taskdir / "task.finished.json").write_text(
        json.dumps(
            {
                "task_id": "D1",
                "task_key": "K1",
                "model_version": "m1",
                "status": "completed",
                "result_checksum": file_checksum(path),
            }
        )
    )
    prior = deepcopy(incoming)
    prior.update(
        upload_operation_id="old-approved-operation",
        input_path=str(taskdir / "task.json"),
        result_path=str(path),
    )
    state = {"tasks": [prior], "processed_task_ids": ["D1"]}
    answer = RemoteBatchRunner(tmp_path / "unused", worker_command=[]).collect_results_with_report(
        state
    )
    assert not answer["results"]
    assert answer["report"]["invalid_count"] == 0
    historical = answer["report"]["metadata_refresh_report"]
    assert historical["invalid_count"] == 1
    assert historical["details_preview"][0]["reason"] == "dft_refresh_upload_operation_id_mismatch"
