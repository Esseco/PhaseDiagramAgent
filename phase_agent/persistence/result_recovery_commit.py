"""Compare-and-swap recovery commit with backups and a stale-writer revision barrier."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import os
import tempfile
from uuid import uuid4

from filelock import FileLock
from phase_agent.tools.state.workflow_control import control_path, read_control, FIELD
from phase_agent.tools.step_runner.file_protocol import _write_json


def _atomic_restore(path, content):
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".rollback-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def commit_result_recovery(state_path, desired_state, files, *, expected_hashes, evidence):
    state_path = Path(state_path).resolve()
    base = state_path.parent
    if not evidence or state_path.name != "state.json":
        raise ValueError("recovery_commit_requires_evidence_and_canonical_state")
    payloads = {Path(k).resolve(): v for k, v in files.items()}
    if state_path in payloads or any(not p.is_relative_to(base / "ledgers") for p in payloads):
        raise ValueError("recovery_commit_requires_project_ledgers")
    revision = uuid4().hex
    desired_state = {**desired_state, FIELD: revision}
    payloads[state_path] = desired_state
    if set(map(str, payloads)) != set(expected_hashes):
        raise ValueError("recovery_commit_requires_all_file_hashes")
    backup = base / "backups" / ("result-recovery-" + revision)
    control = control_path(state_path)
    with FileLock(str(control) + ".lock", timeout=10):
        for path in payloads:
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if digest != expected_hashes[str(path)]:
                raise ValueError("recovery_commit_source_changed:" + str(path))
        current = json.loads(state_path.read_text(encoding="utf-8"))
        if any(
            current.get(k) != desired_state.get(k)
            for k in ("active_model_version", "confirmed_config_version")
        ):
            raise ValueError("recovery_commit_binding_changed")
        previous = read_control(state_path)
        previous_bytes = control.read_bytes() if control.exists() else None
        backup.mkdir(parents=True)
        for path in payloads:
            target = backup / path.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shutil.copy2(path, target)
        if previous_bytes is not None:
            (backup / control.name).write_bytes(previous_bytes)
        transaction_path = backup / "transaction.json"
        targets = []
        for path, value in payloads.items():
            staged = backup / "prepared" / path.relative_to(base)
            _write_json(staged, value)
            targets.append(
                {
                    "relative_path": path.relative_to(base).as_posix(),
                    "before_sha256": expected_hashes[str(path)],
                    "prepared_path": str(staged.relative_to(backup)),
                    "desired_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                }
            )
        journal = {
            "revision": revision,
            "state_path": str(state_path),
            "status": "prepared",
            "targets": targets,
        }
        _write_json(transaction_path, journal)
        record = {
            "schema_version": 1,
            "state_path": str(state_path),
            "revision": revision,
            "previous_revision": previous.get("revision") if previous else None,
            "binding": {
                k: desired_state.get(k)
                for k in ("active_model_version", "confirmed_config_version")
            },
            "patch": deepcopy((previous or {}).get("patch") or {}),
            "evidence": deepcopy(evidence),
            "restore_allowed": False,
            "result_recovery_transaction": str(transaction_path),
        }
        changed = []
        try:
            _write_json(control, record)
            _write_json(transaction_path, {**journal, "status": "committing"})
            for path, value in payloads.items():
                _write_json(path, value)
                changed.append(path)
        except Exception:
            # All originals remain in the backup; restore only files this transaction replaced.
            for path in changed:
                original = backup / path.relative_to(base)
                if original.exists():
                    _atomic_restore(path, original.read_bytes())
                else:
                    path.unlink(missing_ok=True)
            if previous_bytes is not None:
                _atomic_restore(control, previous_bytes)
            else:
                control.unlink(missing_ok=True)
            _write_json(transaction_path, {**journal, "status": "rolled_back"})
            raise
        _write_json(transaction_path, {**journal, "status": "completed"})
        _write_json(
            backup / "commit.json",
            {
                "revision": revision,
                "evidence": evidence,
                "committed_files": [str(p) for p in payloads],
                "status": "completed",
            },
        )
    return {
        "status": "completed",
        "revision": revision,
        "backup": str(backup),
        "state": desired_state,
    }


def recovery_transaction_status(state_path):
    """Inspect without opening or rewriting the possibly interrupted scientific state."""
    state_path = Path(state_path).resolve()
    control = read_control(state_path)
    if not control or not control.get("result_recovery_transaction"):
        return {"status": "none"}
    path = Path(control["result_recovery_transaction"]).resolve()
    if not path.is_relative_to(state_path.parent / "backups"):
        raise ValueError("recovery_transaction_outside_project")
    journal = json.loads(path.read_text(encoding="utf-8"))
    if journal.get("revision") != control["revision"] or journal.get("state_path") != str(
        state_path
    ):
        raise ValueError("recovery_transaction_owner_mismatch")
    return {
        "status": "completed" if journal["status"] == "completed" else "interrupted",
        "revision": control["revision"],
        "journal": str(path),
        "evidence": control.get("evidence"),
        "submitted": False,
    }


def resume_result_recovery(state_path, *, revision):
    """Finish only the staged, previously authorized result commit; never replay tools."""
    state_path = Path(state_path).resolve()
    with FileLock(str(control_path(state_path)) + ".lock", timeout=10):
        status = recovery_transaction_status(state_path)
        if status.get("revision") != revision:
            raise ValueError("recovery_transaction_revision_changed")
        if status["status"] == "completed":
            return {**status, "idempotent_replay": True}
        journal_path = Path(status["journal"])
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        payloads = []
        for row in journal["targets"]:
            target = (state_path.parent / row["relative_path"]).resolve()
            prepared = (journal_path.parent / row["prepared_path"]).resolve()
            if (
                target != state_path and not target.is_relative_to(state_path.parent / "ledgers")
            ) or not prepared.is_relative_to(journal_path.parent / "prepared"):
                raise ValueError("recovery_transaction_unsafe_target")
            if hashlib.sha256(prepared.read_bytes()).hexdigest() != row["desired_sha256"]:
                raise ValueError("recovery_transaction_staged_checksum_mismatch")
            current = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
            if current not in {row["before_sha256"], row["desired_sha256"]}:
                raise ValueError("recovery_transaction_source_changed")
            payloads.append((target, json.loads(prepared.read_text(encoding="utf-8"))))
        for target, payload in payloads:
            _write_json(target, payload)
        _write_json(journal_path, {**journal, "status": "completed"})
        return {"status": "completed", "revision": revision, "submitted": False}


def refresh_manager_after_recovery(manager, state, ledger_path):
    """A long-lived handler must not save a pre-recovery manager over the new ledger."""
    revision = state.get(FIELD)
    if (
        not revision
        or not ledger_path
        or getattr(manager, "_workflow_control_revision", None) == revision
    ):
        return False
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager

    if not isinstance(manager, PhaseDataManager):
        return False
    path = Path(ledger_path)
    if not path.is_file():
        raise ValueError("recovery_manager_ledger_missing")
    loaded = PhaseDataManager.load(path)
    if manager.boundary != loaded.boundary:
        raise ValueError("recovery_manager_boundary_changed")
    manager.data = loaded.data
    manager._workflow_control_revision = revision
    return True
