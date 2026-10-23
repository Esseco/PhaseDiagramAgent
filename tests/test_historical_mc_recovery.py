"""Collect-only historical configuration drift never weakens task or model identity."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import pytest

from tests.test_collect_only_recovery import fixture, build
from phase_agent.tools.relax_mc_batch.recovery_receipt import (
    register_recovery,
    selected_recovery,
    validate_recovery_scope,
)
from phase_agent.tools.relax_mc_batch.handoff import result_archive
from phase_agent.tools.relax_mc_batch.results import import_results, recover_available


def setup(tmp_path):
    original, evidence, job = fixture(tmp_path)
    result = build(tmp_path, original, evidence, job)
    directory = tmp_path / "batch"
    directory.mkdir()
    output = tmp_path / "batch_output"
    output.mkdir()
    shutil.copy2(original, output / "results.zip")
    recovery = tmp_path / "batch_recovery" / "verified"
    recovery.mkdir(parents=True)
    archive = recovery / "recovered.results.zip"
    shutil.copy2(result["archive"], archive)
    task = json.loads((tmp_path / "original/workflow_state.json").read_text())["tasks"][0]
    record = {
        "bundle_id": "b",
        "plan_checksum": "approved",
        "model_version": "m",
        "config_version": "c",
        "directory": str(directory),
        "expected_results_archive": str(output / "results.zip"),
        "branch_ids": ["b1"],
    }
    state = {
        "tasks": [task],
        "active_model_version": "m",
        "confirmed_config_version": "new",
        "relax_mc_batches": {"b": record},
    }
    return record, state, archive


def test_register_preserves_state_and_original_and_selects_archive(tmp_path):
    record, state, archive = setup(tmp_path)
    before = deepcopy(state)
    original = result_archive(record).read_bytes()
    register_recovery(record, state, archive)
    assert state == before
    assert result_archive(record) == archive
    assert (tmp_path / "batch_output/results.zip").read_bytes() == original
    assert register_recovery(record, state, archive)["submitted"] is False


@pytest.mark.parametrize(
    "key",
    [
        "task_key",
        "task_checksum",
        "structure_id",
        "branch_id",
        "model_version",
        "config_version",
        "stage",
        "batch_id",
    ],
)
def test_reject_changed_registration_without_publishing_pointer(tmp_path, key):
    record, state, archive = setup(tmp_path)
    state["tasks"][0][key] = "changed"
    with pytest.raises(ValueError, match="binding_mismatch"):
        register_recovery(record, state, archive)
    assert not (Path(record["directory"]) / "recovery_return.json").exists()


@pytest.mark.parametrize("mode", ["cancelled", "missing"])
def test_reject_cancelled_or_unregistered_task(tmp_path, mode):
    record, state, archive = setup(tmp_path)
    if mode == "missing":
        state["tasks"] = []
    else:
        state["tasks"][0]["recovery_cancelled"] = True
    with pytest.raises(ValueError, match="registered_active_task"):
        register_recovery(record, state, archive)


def test_corrupt_pointer_is_reported_without_import(tmp_path):
    record, state, archive = setup(tmp_path)
    register_recovery(record, state, archive)
    archive.write_bytes(b"corrupt")
    result = recover_available(state, {})
    assert (
        result["relax_mc_batches"]["b"]["return_diagnostic"]["code"] == "invalid_recovery_receipt"
    )
    assert result["tasks"] == state["tasks"]


def test_original_change_rejected(tmp_path):
    record, state, archive = setup(tmp_path)
    register_recovery(record, state, archive)
    (tmp_path / "batch_output/results.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="original_archive_changed"):
        validate_recovery_scope(archive, record, state)


def test_model_switch_remains_rejected(tmp_path):
    record, state, archive = setup(tmp_path)
    register_recovery(record, state, archive)
    state["active_model_version"] = "other-model"
    with pytest.raises(ValueError, match="batch_model_or_config_version_mismatch"):
        import_results(
            action={"parameters": {"bundle_id": "b"}},
            context={"event_state": state, "effective_config": {}, "config_version": "new"},
        )


def test_configuration_drift_imports_and_replay_does_not_settle_again(tmp_path, monkeypatch):
    from importlib import import_module

    module = import_module("phase_agent.tools.relax_mc_batch.results")
    record, state, archive = setup(tmp_path)
    register_recovery(record, state, archive)

    class Manager:
        data = {"structures": {"s1": {"branch_id": "b1"}}}

    # Exercise real verification and reconciliation; isolate geometry-dependent feedback.
    monkeypatch.setattr(
        module, "apply_scientific_feedback", lambda state, *a, **k: {"state": state}
    )
    analysis = import_module("phase_agent.analysis.phase.phase_round_analysis")
    monkeypatch.setattr(
        analysis, "publish_relax_mc_analysis", lambda *a, **k: {"status": "completed"}
    )
    quality = import_module("phase_agent.analysis.phase.phase_quality_reports")
    monkeypatch.setattr(quality, "candidate_quality_evidence", lambda state, rows: rows)
    ctx = {
        "event_state": state,
        "effective_config": {"upload_batches_directory": str(tmp_path / "uploads"), "mlip": {}},
        "config_version": "new",
        "manager": Manager(),
    }
    recovered = recover_available(state, ctx)
    assert "return_error" not in recovered["relax_mc_batches"]["b"]
    result = {"state": recovered}
    assert result["state"]["tasks"][0]["status"] == "completed"
    assert result["state"]["confirmed_config_version"] == "new"
    result["state"].setdefault("budget_usage", {})
    first = deepcopy(result["state"])
    replay = import_results(
        action={"parameters": {"bundle_id": "b"}}, context={**ctx, "event_state": first}
    )
    assert replay["status"] == "already_imported"
    assert replay["state"]["budget_usage"] == first["budget_usage"]
    assert replay["state"]["processed_task_ids"] == first["processed_task_ids"]


@pytest.mark.parametrize("status", ["cancelled", "superseded"])
def test_cancelled_batch_registration_is_rejected(tmp_path, status):
    record, state, archive = setup(tmp_path)
    record["status"] = status
    with pytest.raises(ValueError, match="manual_result_review"):
        register_recovery(record, state, archive)


def test_registration_rejects_inactive_model(tmp_path):
    record, state, archive = setup(tmp_path)
    state["active_model_version"] = "other"
    with pytest.raises(ValueError, match="active_model_mismatch"):
        register_recovery(record, state, archive)


def test_cli_registers_pointer_without_rewriting_state(tmp_path):
    from phase_agent.tools.relax_mc_batch.cli import main

    record, state, archive = setup(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    before = state_path.read_bytes()
    assert (
        main(
            [
                "register-recovery",
                "--state",
                str(state_path),
                "--bundle-id",
                "b",
                "--archive",
                str(archive),
            ]
        )
        == 0
    )
    assert selected_recovery(record) == archive
    assert state_path.read_bytes() == before


def test_registration_rejects_unrelated_batch_binding(tmp_path):
    record, state, archive = setup(tmp_path)
    record["plan_checksum"] = "another-plan"
    with pytest.raises(ValueError, match="original_batch_binding_mismatch"):
        register_recovery(record, state, archive)


def test_registration_rejects_archive_outside_batch(tmp_path):
    record, state, archive = setup(tmp_path)
    outside = tmp_path / "outside.zip"
    shutil.copy2(archive, outside)
    with pytest.raises(ValueError, match="outside_batch"):
        register_recovery(record, state, outside)
