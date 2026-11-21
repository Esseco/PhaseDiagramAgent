"""Repair lost relocation metadata without replacing current control authority."""

import hashlib
import json
from pathlib import Path

from filelock import FileLock

from phase_agent.tools.state.workflow_control import control_path, read_control
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def restore_relocation_metadata(state_path, *, backup_path, backup_sha256, expected_revision):
    state_path = Path(state_path).resolve()
    root = state_path.parent.parent
    backup = Path(backup_path).resolve()
    if not any(
        backup.is_relative_to(parent)
        for parent in (root / "history_backups", state_path.parent / "backups")
    ) or backup.name != "state.json.control.json":
        raise ValueError("invalid_relocation_backup_path")
    raw = backup.read_bytes()
    if hashlib.sha256(raw).hexdigest() != backup_sha256:
        raise ValueError("relocation_backup_digest_mismatch")
    old = json.loads(raw)
    control = control_path(state_path)
    with FileLock(str(state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10):
        with FileLock(str(control) + ".lock", timeout=10):
            current = read_control(state_path)
            if not current or current["revision"] != expected_revision:
                raise ValueError("relocation_current_revision_changed")
            state = read_json(state_path)
            version = state.get("path_relocation_revision")
            moves = old.get("path_relocations")
            if (
                not version or old.get("path_relocation_revision") != version
                or old.get("state_path") != str(state_path)
                or not isinstance(moves, list) or not moves
                or state.get("workflow_control_revision") != expected_revision
            ):
                raise ValueError("relocation_backup_not_bound_to_current_state")
            for move in moves:
                if not all(Path(move[key]).resolve().is_relative_to(root) for key in ("from", "to")):
                    raise ValueError("relocation_backup_scope_invalid")
            fields = {"path_relocation_revision": version, "path_relocations": moves}
            if any(key in current for key in fields):
                if any(current.get(key) != value for key, value in fields.items()):
                    raise ValueError("relocation_metadata_already_conflicts")
                status = "already_present"
            else:
                write_json(control, {**current, **fields})
                status = "restored"
            return {"status": status, "revision": expected_revision, "relocation_count": len(moves)}
