from pathlib import Path
import hashlib
import json

import pytest

from phase_agent.persistence.result_recovery_commit import commit_result_recovery
from phase_agent.tools.state.workflow_control import state_write_guard, restore_control


def setup(tmp_path):
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledgers/phase_data.json"
    ledger.parent.mkdir()
    old = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "tasks": [{"status": "pending"}],
    }
    state.write_text(json.dumps(old))
    ledger.write_text(json.dumps({"structures": {"s": {}}}))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (state, ledger)}
    return state, ledger, old, hashes


def test_recovery_commit_has_backup_and_blocks_stale_writers_and_restore(tmp_path):
    state, ledger, old, hashes = setup(tmp_path)
    new = {**old, "tasks": [{"status": "completed"}]}
    result = commit_result_recovery(
        state,
        new,
        {ledger: {"structures": {"s": {}, "new": {}}}},
        expected_hashes=hashes,
        evidence={"job": "1"},
    )
    assert json.loads(state.read_text())["tasks"] == new["tasks"]
    assert (tmp_path / "backups" / Path(result["backup"]).name / "state.json").is_file()
    with pytest.raises(ValueError, match="stale_workflow"):
        with state_write_guard(state, old):
            pass
    with pytest.raises(ValueError, match="stale_business_state"):
        restore_control(state, old)


def test_changed_ledger_aborts_before_any_state_or_control_write(tmp_path):
    state, ledger, old, hashes = setup(tmp_path)
    ledger.write_text("changed")
    with pytest.raises(ValueError, match="source_changed"):
        commit_result_recovery(
            state, old, {ledger: {}}, expected_hashes=hashes, evidence={"job": "1"}
        )
    assert json.loads(state.read_text()) == old
    assert not state.with_name("state.json.control.json").exists()


def test_failed_second_file_publish_restores_previous_files_and_control(tmp_path, monkeypatch):
    from phase_agent.persistence import result_recovery_commit as module

    state, ledger, old, hashes = setup(tmp_path)
    original = {p: p.read_bytes() for p in (state, ledger)}
    write = module._write_json

    def fail(path, value):
        if Path(path) == state:
            raise OSError("disk error")
        return write(path, value)

    monkeypatch.setattr(module, "_write_json", fail)
    with pytest.raises(OSError, match="disk error"):
        commit_result_recovery(
            state, old, {ledger: {"changed": True}}, expected_hashes=hashes, evidence={"job": "1"}
        )
    assert all(p.read_bytes() == data for p, data in original.items())
    assert not state.with_name("state.json.control.json").exists()


def test_process_crash_between_ledger_and_state_resumes_only_staged_facts(tmp_path, monkeypatch):
    from phase_agent.persistence import result_recovery_commit as module
    from phase_agent.tools.state.workflow_control import read_control

    state, ledger, old, hashes = setup(tmp_path)
    write = module._write_json

    class SimulatedProcessCrash(BaseException):
        pass

    def crash(path, value):
        if Path(path) == state:
            raise SimulatedProcessCrash()
        return write(path, value)

    monkeypatch.setattr(module, "_write_json", crash)
    with pytest.raises(SimulatedProcessCrash):
        commit_result_recovery(
            state,
            {**old, "tasks": [{"status": "completed"}]},
            {ledger: {"recovered": True}},
            expected_hashes=hashes,
            evidence={"original_approval": "verified", "job": "1"},
        )
    status = module.recovery_transaction_status(state)
    assert status["status"] == "interrupted"
    revision = read_control(state)["revision"]
    monkeypatch.setattr(module, "_write_json", write)
    result = module.resume_result_recovery(state, revision=revision)
    assert result["submitted"] is False and result["status"] == "completed"
    assert json.loads(ledger.read_text()) == {"recovered": True}
    assert json.loads(state.read_text())["tasks"] == [{"status": "completed"}]
    assert module.resume_result_recovery(state, revision=revision)["idempotent_replay"]


def test_recovery_refuses_concurrent_edit_after_interrupted_commit(tmp_path, monkeypatch):
    from phase_agent.persistence import result_recovery_commit as module

    state, ledger, old, hashes = setup(tmp_path)
    write = module._write_json

    class Crash(BaseException):
        pass

    def crash(path, value):
        if Path(path) == state:
            raise Crash()
        return write(path, value)

    monkeypatch.setattr(module, "_write_json", crash)
    with pytest.raises(Crash):
        commit_result_recovery(
            state, old, {ledger: {"updated": 1}}, expected_hashes=hashes, evidence={"job": "1"}
        )
    revision = module.recovery_transaction_status(state)["revision"]
    ledger.write_text('{"edited":true}')
    monkeypatch.setattr(module, "_write_json", write)
    with pytest.raises(ValueError, match="source_changed"):
        module.resume_result_recovery(state, revision=revision)
    assert json.loads(ledger.read_text()) == {"edited": True}


def test_long_lived_manager_reloads_committed_ledger_once_per_revision(tmp_path):
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.persistence.result_recovery_commit import refresh_manager_after_recovery

    boundary = {
        "P": ["O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"Fe": 1},
    }
    manager = PhaseDataManager(boundary)
    path = tmp_path / "ledger.json"
    manager.data["recovery_fact"] = "old"
    fresh = PhaseDataManager(boundary)
    fresh.data["recovery_fact"] = "new"
    fresh.save(path)
    identity = id(manager)
    assert refresh_manager_after_recovery(manager, {"workflow_control_revision": "r1"}, path)
    assert id(manager) == identity and manager.data["recovery_fact"] == "new"
    assert not refresh_manager_after_recovery(manager, {"workflow_control_revision": "r1"}, path)



@pytest.mark.parametrize("extended_posix", [False, True])
def test_extended_drive_paths_relocate_without_touching_approved_snapshots(tmp_path, extended_posix):
    from copy import deepcopy
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths
    from phase_agent.tools.policy.file_approval import proposal_hash

    old = tmp_path / "submissions/verified/batch"
    new = tmp_path / "submissions/epoch1/verified_results/batch"
    native = "\\\\?\\" + str(old / "initial_relaxed.vasp")
    source = native.replace(chr(92), "/") if extended_posix else native
    unmapped = chr(92) * 2 + "?" + chr(92) + str(tmp_path / "elsewhere/unmapped.vasp")
    proposal = {"raw_action": {"parameters": {"structure_path": source}}}
    original = {"structure_path": source, "outside": unmapped, "energy": -123.456,
                "confirmed_snapshot": {"structure_path": source}, "agent_proposal": proposal,
                "action_records": [{"final_action": {"structure_path": source}}]}
    before = deepcopy(original)
    changed = relocate_workspace_paths(original, [{"from": str(old), "to": str(new)}], tmp_path, preserve_snapshots=True)
    assert Path(changed["structure_path"]) == new / "initial_relaxed.vasp"
    assert changed["outside"] == unmapped and changed["energy"] == -123.456
    assert changed["confirmed_snapshot"] == original["confirmed_snapshot"]
    assert proposal_hash(changed["agent_proposal"]) == proposal_hash(proposal)
    assert changed["action_records"] == original["action_records"] and original == before


def test_recovery_commit_keeps_existing_path_relocation_control(tmp_path):
    from phase_agent.tools.state.workflow_control import control_path, read_control

    state, ledger, old, hashes = setup(tmp_path)
    moves = [{"from": str(tmp_path / "old"), "to": str(tmp_path / "new")}]
    control_path(state).write_text(json.dumps({"state_path": str(state.resolve()),
        "revision": "prior", "patch": {}, "path_relocations": moves,
        "path_relocation_revision": "paths-prior"}), encoding="utf-8")
    commit_result_recovery(state, old, {ledger: {"structures": {}}}, expected_hashes=hashes,
                           evidence={"maintenance": "verified_path_repair"})
    saved = read_control(state)
    assert saved["path_relocations"] == moves
    assert saved["path_relocation_revision"] == "paths-prior"
    assert saved["restore_allowed"] is False
