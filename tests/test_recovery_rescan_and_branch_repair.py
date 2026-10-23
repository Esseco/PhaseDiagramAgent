"""No real calculations, service control, or scientific project writes."""

from copy import deepcopy
from importlib import import_module

import pytest

from tests.test_on_demand_branch import context, review


@pytest.mark.parametrize("wire", ["tool", "action_type", "envelope", "operation"])
def test_branch_correction_keeps_original_tool_contract(wire):
    from phase_agent.decisions.agent.branch_review_repair import repair_branch_review
    from phase_agent.tools.dispatch.scientific_operations import with_operation_contracts

    valid = review()
    valid["parameters"].update(sampling_mode="region_adaptive", defer_reason="暂缓")
    invalid = deepcopy(valid)
    invalid["branch_pool_review"]["evidence_refs"] = ["invented"]
    if wire == "action_type":
        invalid["action_type"] = invalid.pop("tool")
    before = deepcopy(invalid)
    calls = []

    def client(payload):
        calls.append(payload)
        assert payload["allowed_tools"] == ["generate_branches"]
        assert {row["tool"] for row in payload["scientific_operations"]} == {"generate_branches"}
        corrected = deepcopy(valid)
        if wire in {"action_type", "envelope"}:
            corrected["action_type"] = corrected.pop("tool")
        elif wire == "operation":
            corrected["operation"] = corrected.pop("tool")
        if wire == "envelope":
            return {"action": corrected, "_llm_usage": {"calls": 1}}
        return corrected

    payload = with_operation_contracts(
        {
            "decision_context": context(),
            "allowed_tools": ["generate_branches", "prepare_local_batch_files"],
        }
    )
    result = repair_branch_review(client, payload, invalid)
    assert result["tool"] == "generate_branches"
    assert result["branch_pool_review"]["evidence_refs"] == ["current"]
    assert len(calls) == 1 and invalid == before
    if wire == "envelope":
        assert result["_llm_usage"] == {"calls": 1}


@pytest.mark.parametrize("changed", ["prepare_local_batch_files", "pause_search", "dialogue"])
def test_branch_correction_still_rejects_tool_change(changed):
    from phase_agent.decisions.agent.branch_review_repair import repair_branch_review

    invalid = review()
    invalid["branch_pool_review"]["evidence_refs"] = ["invented"]
    corrected = {"tool": changed} if changed != "dialogue" else {"kind": "answer", "answer": "ok"}
    with pytest.raises(ValueError, match="branch_review_repair_must_not_change_the_tool"):
        repair_branch_review(lambda payload: corrected, {"decision_context": context()}, invalid)


@pytest.mark.parametrize("change", ["config", "model"])
@pytest.mark.parametrize("archive_state", ["same", "changed", "unimported"])
def test_historical_rescan_only_acknowledges_identical_imported_archive(
    tmp_path, monkeypatch, change, archive_state
):
    module = import_module("phase_agent.tools.relax_mc_batch.results")
    archive = tmp_path / "results.zip"
    archive.write_bytes(b"already verified old return")
    digest = module.checksum(archive)
    record = {
        "status": "failed",
        "model_version": "m1",
        "config_version": "c1",
        "expected_results_archive": str(archive),
        "import_checksum": digest,
        "result_summary": {"status": "failed", "task_count": 166},
        "return_error": "batch_model_or_config_version_mismatch; inspect historical results",
        "return_diagnostic": {"error": "batch_model_or_config_version_mismatch"},
    }
    if archive_state == "changed":
        archive.write_bytes(b"a different late return")
    elif archive_state == "unimported":
        record.pop("import_checksum")
    archive_digest = module.checksum(archive)
    state = {
        "active_model_version": "m2" if change == "model" else "m1",
        "tasks": [{"task_id": "already-computed", "status": "completed"}],
        "budget_usage": {"cost": 12},
        "qbc_candidates": [{"candidate_id": "existing"}],
        "relax_mc_batches": {"b": record},
    }
    before = deepcopy(state)
    driver = {
        "origin": "remote_controller",
        "code": "remote_driver_failed",
        "error": "squeue failed",
    }
    monkeypatch.setattr(module, "_driver_diagnostic", lambda record: deepcopy(driver))

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical rescan must not extract, activate, or settle results")

    monkeypatch.setattr(module, "import_bundle", forbidden)
    monkeypatch.setattr(module, "apply_scientific_feedback", forbidden)
    ctx = {"effective_config": {"mlip": {}}, "config_version": "c2" if change == "config" else "c1"}
    result = module.recover_available(state, ctx)
    assert state == before
    recovered = result["relax_mc_batches"]["b"]
    assert recovered["driver_diagnostic"] == driver
    assert recovered["result_summary"] == before["relax_mc_batches"]["b"]["result_summary"]
    assert recovered["status"] == "failed"
    for key in ("tasks", "budget_usage", "qbc_candidates", "active_model_version"):
        assert result[key] == before[key]
    if archive_state == "same":
        assert "return_error" not in recovered
        assert "return_diagnostic" not in recovered
    else:
        assert "batch_model_or_config_version_mismatch" in recovered["return_error"]
    assert module.checksum(archive) == archive_digest


def test_historical_receipt_updates_latest_state_after_another_import(tmp_path, monkeypatch):
    module = import_module("phase_agent.tools.relax_mc_batch.results")
    archive = tmp_path / "results.zip"
    archive.write_bytes(b"old return")
    historical = {
        "model_version": "old",
        "config_version": "c1",
        "status": "completed",
        "expected_results_archive": str(archive),
        "import_checksum": module.checksum(archive),
        "return_error": "batch_model_or_config_version_mismatch",
    }
    state = {
        "active_model_version": "new",
        "relax_mc_batches": {
            "active": {**historical, "model_version": "new", "config_version": "c2"},
            "historical": historical,
        },
    }
    monkeypatch.setattr(module, "_driver_diagnostic", lambda record: None)

    def imported(**kwargs):
        assert kwargs["action"]["parameters"]["bundle_id"] == "active"
        return {"state": deepcopy(kwargs["context"]["event_state"])}

    monkeypatch.setattr(module, "import_results", imported)
    result = module.recover_available(state, {"config_version": "c2"})
    assert "return_error" not in result["relax_mc_batches"]["historical"]
    assert "return_error" in state["relax_mc_batches"]["historical"]


def test_explicit_import_keeps_historical_binding_guard(tmp_path):
    from phase_agent.tools.relax_mc_batch.results import import_results
    from phase_agent.tools.remote.transfer_bundle import checksum

    archive = tmp_path / "results.zip"
    archive.write_bytes(b"old return")
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "config_version": "c1",
                "expected_results_archive": str(archive),
                "import_checksum": checksum(archive),
            }
        },
    }
    with pytest.raises(ValueError, match="batch_model_or_config_version_mismatch"):
        import_results(
            action={"parameters": {"bundle_id": "b"}},
            context={
                "event_state": state,
                "effective_config": {},
                "config_version": "c2",
            },
        )
