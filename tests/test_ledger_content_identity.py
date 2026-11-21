"""Ledger identity detects another process even when metadata is preserved."""

import subprocess
import sys

import pytest

from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager, refresh_if_changed


def _ledger(tmp_path):
    boundary = {
        "P": ["O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"Fe": 1},
    }
    path = tmp_path / "phase.json"
    manager = PhaseDataManager(boundary)
    manager.data["audit"] = "A"
    manager.save(path)
    return path, PhaseDataManager.load(path)


def _other_process_same_metadata(path):
    before = path.stat()
    code = (
        "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); st=p.stat(); "
        'b=p.read_bytes(); old=b\'\\"audit\\": \\"A\\"\'; new=b\'\\"audit\\": \\"B\\"\'; '
        "assert old in b; p.write_bytes(b.replace(old,new)); os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    )


def test_stale_ledger_writer_cannot_overwrite_same_metadata_change(tmp_path):
    path, manager = _ledger(tmp_path)
    _other_process_same_metadata(path)
    manager.data["local_edit"] = True
    with pytest.raises(ValueError, match="ledger_changed_since_load"):
        manager.save(path)
    assert PhaseDataManager.load(path).data["audit"] == "B"
    assert "local_edit" not in PhaseDataManager.load(path).data


def test_refresh_adopts_same_metadata_change_only_when_local_state_clean(tmp_path):
    path, manager = _ledger(tmp_path)
    _other_process_same_metadata(path)
    assert refresh_if_changed(manager, path) is True
    assert manager.data["audit"] == "B"
    manager.data["local_edit"] = True
    before = path.stat()
    code = (
        "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); st=p.stat(); "
        'b=p.read_bytes(); p.write_bytes(b.replace(b\'\\"audit\\": \\"B\\"\',b\'\\"audit\\": \\"C\\"\')); '
        "os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    assert before.st_size == path.stat().st_size and before.st_mtime_ns == path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="ledger_memory_changed_requires_reconciliation"):
        refresh_if_changed(manager, path)
    assert manager.data["local_edit"] is True
    assert PhaseDataManager.load(path).data["audit"] == "C"


def test_change_during_serialization_is_not_overwritten(tmp_path, monkeypatch):
    import phase_agent.persistence.ledger.phase_data_manager as module

    path, manager = _ledger(tmp_path)
    manager.data["local_edit"] = True
    original = module.json.dump

    def competing(value, stream, *args, **kwargs):
        result = original(value, stream, *args, **kwargs)
        _other_process_same_metadata(path)
        return result

    monkeypatch.setattr(module.json, "dump", competing)
    with pytest.raises(ValueError, match="ledger_changed_during_save"):
        manager.save(path)
    saved = PhaseDataManager.load(path).data
    assert saved["audit"] == "B" and "local_edit" not in saved
    assert not list(path.parent.glob("phase.json.*.tmp"))
