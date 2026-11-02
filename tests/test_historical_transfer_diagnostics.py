from copy import deepcopy
import pytest
from phase_agent.tools.workflows.recover_saved_results import (
    _classify_transfer_diagnostics, _empty_collection_failure,
)


@pytest.mark.parametrize("processed,status,historical", [
    (True, "completed", True), (True, "failed", True),
    (False, "completed", False), (True, "running", False),
])
def test_dispatch_diagnostic_scope(tmp_path, monkeypatch, processed, status, historical):
    from phase_agent.tools.remote import legacy_recovery
    folder = tmp_path / "old_output"
    monkeypatch.setattr(legacy_recovery, "recovery_handoffs", lambda state: iter([
        {"return_directory": str(folder), "task_ids": ["old"]},
    ]))
    state = {"tasks": [{"task_id": "old", "status": status}],
             "processed_task_ids": ["old"] if processed else []}
    report = {"transfer_errors": [{"archive": str(folder / "diagnostics.zip"),
                                  "reason": "remote_dispatch_failed:old format error"}]}
    before = deepcopy(state)
    checked = _classify_transfer_diagnostics(report, state)
    assert bool(checked["historical_transfer_errors"]) is historical
    assert bool(_empty_collection_failure(checked, state)) is not historical
    assert state == before and report["transfer_errors"]


def test_invalid_archive_and_unbound_diagnostics_still_block(tmp_path, monkeypatch):
    from phase_agent.tools.remote import legacy_recovery
    monkeypatch.setattr(legacy_recovery, "recovery_handoffs", lambda state: iter([]))
    state = {"tasks": [{"task_id": "old", "status": "completed"}],
             "processed_task_ids": ["old"],
             "slurm_batches": [{"task_ids": ["old"], "handoff": {"return_directory": str(tmp_path)}}]}
    for filename, reason in [("new.results.zip", "checksum mismatch"),
                             ("diagnostics.zip", "bad diagnostic"),
                             ("unknown/diagnostics.zip", "remote_dispatch_failed:new")]:
        report = {"transfer_errors": [{"archive": str(tmp_path / filename), "reason": reason}]}
        assert _empty_collection_failure(_classify_transfer_diagnostics(report, state), state)
