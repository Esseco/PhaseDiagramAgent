import hashlib
import json

import pytest

from phase_agent.runtime.restore_relocation_metadata import restore_relocation_metadata


def test_restore_only_current_bound_relocation_fields(tmp_path):
    path = tmp_path / "workflow_state" / "state.json"
    path.parent.mkdir()
    state = {"workflow_control_revision": "current", "path_relocation_revision": "move"}
    path.write_text(json.dumps(state), encoding="utf-8")
    control = path.with_name("state.json.control.json")
    current = {"state_path": str(path), "revision": "current", "patch": {"x": 1}}
    control.write_text(json.dumps(current), encoding="utf-8")
    backup = tmp_path / "history_backups" / control.name
    backup.parent.mkdir()
    fields = {
        "path_relocation_revision": "move",
        "path_relocations": [{"from": str(tmp_path / "old"), "to": str(tmp_path / "new")}],
    }
    backup.write_text(json.dumps({**current, **fields, "revision": "old", "patch": {}}), encoding="utf-8")
    args = {"backup_path": backup, "backup_sha256": hashlib.sha256(backup.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="revision_changed"):
        restore_relocation_metadata(path, **args, expected_revision="wrong")
    assert json.loads(control.read_bytes()) == current
    assert restore_relocation_metadata(path, **args, expected_revision="current")["status"] == "restored"
    assert json.loads(control.read_bytes()) == {**current, **fields}
    assert json.loads(path.read_bytes()) == state
    assert restore_relocation_metadata(path, **args, expected_revision="current")["status"] == "already_present"
