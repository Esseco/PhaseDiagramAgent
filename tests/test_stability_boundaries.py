import json
from copy import deepcopy
from pathlib import Path
import threading

import pytest

from phase_agent.runtime.ui_operations import UIOperations
from phase_agent.tools.remote.integrity import verify_result
from phase_agent.persistence.ledger.collect_calculation_results import collect_calculation_results
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


def manager_and_result():
    manager = PhaseDataManager({"P": ["O3"], "H": {"max_det": 2}, "TM_ratio": {"Ni": 1}})
    bid = manager.add_branch(
        P="O3",
        H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        x=1,
        T=["Ni"],
        composition={"Na": 1, "Ni": 1, "O": 2},
    )
    sid = manager.add_structure(branch_id=bid, arrangement={"new": True})
    return manager, {
        "task_id": "new-Ni",
        "task_key": "new-key",
        "structure_id": sid,
        "branch_id": bid,
        "stage": "deep_search",
        "status": "completed",
        "model_version": "heldout",
        "outputs": {
            "energy": -10.0,
            "energy_unit": "eV",
            "composition": {"Na": 1, "Ni": 1, "O": 2},
        },
    }


@pytest.mark.parametrize("fault", ["nan", "inf", "boolean", "unit", "composition", "shape"])
def test_invalid_result_does_not_mutate_ledger(fault):
    manager, result = manager_and_result()
    if fault in {"nan", "inf", "boolean"}:
        result["outputs"]["energy"] = {"nan": float("nan"), "inf": float("inf"), "boolean": True}[
            fault
        ]
    elif fault == "unit":
        result["outputs"]["energy_unit"] = "Hartree"
    elif fault == "composition":
        result["outputs"]["composition"]["Ni"] = -1
    else:
        result["outputs"] = ["invalid"]
    before = deepcopy(manager.data)
    with pytest.raises(ValueError):
        collect_calculation_results(manager, result["structure_id"], result)
    assert manager.data == before


@pytest.mark.parametrize("payload", [[], None, 1, "wrong"])
def test_receipt_wrong_json_shape_is_verification_failure(tmp_path, payload):
    result, marker = tmp_path / "result.json", tmp_path / "task.finished.json"
    result.write_text(json.dumps(payload))
    marker.write_text("{}")
    assert verify_result(result, marker, {})["valid"] is False


def test_corrupt_operation_does_not_hide_other_operations(tmp_path):
    store = UIOperations(tmp_path / "state.json", threading.RLock())
    store.directory.mkdir()
    (store.directory / ("a" * 32 + ".json")).write_text("{broken")
    (store.directory / ("b" * 32 + ".json")).write_text(
        json.dumps({"operation_id": "b" * 32, "status": "completed"})
    )
    rows = store.recent()
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"completed", "unavailable"}
    assert (store.directory / ("a" * 32 + ".json")).read_text() == "{broken"


@pytest.mark.parametrize(
    "key", ["model_version", "config_version", "stage", "structure_id", "task_key"]
)
def test_registered_task_wrong_binding_is_isolated_before_settlement(key):
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results

    manager, result = manager_and_result()
    task = {**result, "status": "pending", "config_version": "config-a"}
    state = {"tasks": [task]}
    incoming = deepcopy(result)
    incoming[key] = "different"
    changed = reconcile_task_results(state, [incoming])
    assert changed["state"]["tasks"] == [task]
    assert changed["reconciled"][0]["status"] == "rejected"
    assert not changed["state"].get("processed_task_ids")
    assert result["task_id"] in changed["state"]["result_validation_errors"]


@pytest.mark.parametrize("bad_outputs", [{"energy": float("nan"), "energy_unit": "eV"}, ["bad"]])
def test_bad_result_feedback_isolated_good_result_recorded_and_raw_evidence_retained(bad_outputs):
    from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback

    manager, good = manager_and_result()
    bad = {**deepcopy(good), "task_id": "bad-chemistry", "outputs": bad_outputs}
    state = {"tasks": [good, bad], "active_model_version": "heldout"}
    answer = apply_scientific_feedback(state, [good, bad], manager=manager)
    assert good["task_id"] in answer["state"]["feedback_processed_task_ids"]
    assert bad["task_id"] not in answer["state"]["feedback_processed_task_ids"]
    assert bad["task_id"] in answer["state"]["scientific_feedback_errors"]
    assert len(answer["recorded"]) == 1
    assert next(t for t in answer["state"]["tasks"] if t["task_id"] == bad["task_id"])[
        "outputs"
    ] is bad_outputs or str(
        next(t for t in answer["state"]["tasks"] if t["task_id"] == bad["task_id"])["outputs"]
    ) == str(bad_outputs)


@pytest.mark.parametrize("payload", [[], None, 1, "wrong"])
def test_marker_wrong_shape_rejected_without_uncaught_exception(tmp_path, payload):
    result, marker = tmp_path / "result.json", tmp_path / "task.finished.json"
    result.write_text('{"status":"completed"}')
    marker.write_text(json.dumps(payload))
    assert verify_result(result, marker, {})["valid"] is False


def test_corrupt_operation_older_than_display_limit_still_blocks_new_execution(tmp_path):
    import os

    store = UIOperations(tmp_path / "state.json", threading.RLock())
    store.directory.mkdir()
    path = store.directory / ("a" * 32 + ".json")
    path.write_text("{broken")
    os.utime(path, (0, 0))
    for n in range(21):
        (store.directory / (format(n, "032x") + ".json")).write_text(
            json.dumps({"operation_id": format(n, "032x"), "status": "completed"})
        )
    assert len(store.recent()) == 20
    calls = []
    with pytest.raises(ValueError, match="receipt_review"):
        store.submit(
            lambda: calls.append(1), request_id="blocked", request={"route": "/phase/propose"}
        )
    assert not calls


def test_finished_worker_with_unwritten_completion_is_interrupted_not_forever_running(
    tmp_path, monkeypatch
):
    import time
    from tests.test_ui_operations import wait
    import phase_agent.runtime.ui_operations as module

    store = UIOperations(tmp_path / "state.json", threading.RLock())
    original = module.write_json
    calls = []

    def fail_final(path, value):
        if value["status"] in {"completed", "blocked"}:
            raise PermissionError("cannot save final")
        return original(path, value)

    monkeypatch.setattr(module, "write_json", fail_final)
    first = store.submit(
        lambda: calls.append(1) or {},
        request_id="lost-completion",
        request={"route": "/phase/propose"},
    )
    answer = wait(store, first["operation_id"])
    assert answer["status"] == "interrupted" and len(calls) == 1
    replay = store.submit(
        lambda: calls.append(2), request_id="lost-completion", request={"route": "/phase/propose"}
    )
    assert replay["idempotent_replay"] and len(calls) == 1


def test_different_key_cannot_bypass_interrupted_effect_review(tmp_path, monkeypatch):
    from tests.test_ui_operations import wait

    store = UIOperations(tmp_path / "state.json", threading.RLock())
    first = store.submit(
        lambda: (_ for _ in ()).throw(TimeoutError()),
        request_id="uncertain",
        request={"route": "/phase/decision"},
    )
    assert wait(store, first["operation_id"])["status"] == "blocked"
    with pytest.raises(ValueError, match="receipt_review"):
        store.submit(
            lambda: pytest.fail("must not replay"),
            request_id="new-key",
            request={"route": "/phase/decision"},
        )


@pytest.mark.parametrize("fault", ["steps", "runtime", "scalar", "status_shape"])
def test_bad_metadata_does_not_stop_following_good_result_or_modify_bad_task(fault):
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results

    manager, good = manager_and_result()
    bad = deepcopy(good)
    bad["task_id"] = "bad-meta"
    bad["task_key"] = "bad-meta-key"
    pending = {**bad, "status": "pending"}
    bad["requested_max_mc_steps"] = 100
    if fault == "steps":
        bad["actual_mc_steps"] = "bad"
    elif fault == "runtime":
        bad["runtime_observation"] = ["bad"]
    elif fault == "status_shape":
        bad["status"] = ["completed"]
    else:
        bad = ["bad"]
    answer = reconcile_task_results(
        {"tasks": [pending, {**good, "status": "pending"}]}, [bad, good]
    )
    assert (
        next(t for t in answer["state"]["tasks"] if t["task_id"] == pending["task_id"]) == pending
    )
    assert good["task_id"] in answer["state"]["processed_task_ids"]
    assert answer["reconciled"][0]["status"] == "rejected"


@pytest.mark.parametrize(
    "key", ["task_key", "structure_id", "stage", "config_version", "model_version"]
)
def test_feedback_cannot_reintroduce_results_rejected_by_reconciliation(key):
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
    from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback

    manager, result = manager_and_result()
    result["config_version"] = "c1"
    pending = {**result, "status": "pending"}
    bad = deepcopy(result)
    bad[key] = "unapproved"
    state = reconcile_task_results({"tasks": [pending]}, [bad])["state"]
    before = deepcopy(manager.data)
    answer = apply_scientific_feedback(state, [bad], manager=manager)
    assert answer["state"]["tasks"] == [pending]
    assert manager.data == before and not answer["recorded"]


def test_historical_invalid_record_cannot_be_reactivated_by_cached_phase():
    from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback
    from phase_agent.analysis.phase.refresh_identified_phases import refresh_identified_phases

    manager, result = manager_and_result()
    result["outputs"]["energy"] = float("nan")
    record = {
        "record_id": "old-bad",
        "source_task_id": result["task_id"],
        "structure_id": result["structure_id"],
        "energy_method": "mlip",
        "model_version": "heldout",
        "source_version": "heldout",
        "energy": -10,
        "energy_unit": "eV",
        "phase": "O3",
        "phase_identification_status": "identified",
        "phase_identification": {"status": "identified"},
        "status": "completed",
    }
    answer = apply_scientific_feedback(
        {"tasks": [result], "phase_records": [record], "active_model_version": "heldout"},
        [],
        manager=manager,
    )
    state = answer["state"]
    refresh_identified_phases(state)
    assert state["phase_records"][0]["status"] == "excluded_invalid_result"
    assert not state["phase_records"][0]["checks_passed"]
    assert not (state.get("phase_diagrams", {}).get("mlip") or {}).get("entries")


@pytest.mark.parametrize(
    "key", ["task_key", "structure_id", "stage", "config_version", "model_version"]
)
def test_explicit_null_cannot_erase_registered_binding(key):
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results

    manager, result = manager_and_result()
    result["config_version"] = "c1"
    pending = {**result, "status": "pending"}
    bad = deepcopy(result)
    bad[key] = None
    answer = reconcile_task_results({"tasks": [pending]}, [bad])
    assert answer["state"]["tasks"] == [pending] and answer["reconciled"][0]["status"] == "rejected"


def test_missing_stage_cannot_skip_registered_stage_unit_validation():
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results

    manager, result = manager_and_result()
    pending = {**result, "status": "pending"}
    bad = deepcopy(result)
    bad.pop("stage")
    bad["outputs"]["energy_unit"] = "Hartree"
    answer = reconcile_task_results({"tasks": [pending]}, [bad])
    assert answer["state"]["tasks"] == [pending] and answer["reconciled"][0]["status"] == "rejected"
