"""DFT package repair survives a concurrent state change without overwriting it."""

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.remote.input_validation import validate_handoff
from tests.test_dft_handoff_repair import fixture


def test_repair_conflict_preserves_package_for_verified_retry(tmp_path, monkeypatch):
    import phase_agent.tools.remote.dft_handoff_repair as module
    import phase_agent.tools.state.execution_receipts as receipts

    root, state, old, digest = fixture(tmp_path)
    path = root / "workflow_state" / "state.json"
    path.parent.mkdir()
    write_json(path, state)
    monkeypatch.setattr(receipts, "recovery_report", lambda *_: {"unsettled": []})
    original = module.repair_dft_handoff
    published = []
    first = True

    def competing(*args, **kwargs):
        nonlocal first
        result = original(*args, **kwargs)
        published.append(Path(result["handoff"]["folder"]))
        if first:
            first = False
            script = (
                "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
                "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
            )
            subprocess.run([sys.executable, "-c", script, str(path)], check=True)
        return result

    monkeypatch.setattr(module, "repair_dft_handoff", competing)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.repair_dft_handoff("decision", digest, 2)
    saved = read_json(path)
    assert saved["other_process_update"] is True
    assert saved["tasks"] == state["tasks"] and not saved.get("dft_handoff_repairs")
    assert all(row["handoff"]["folder"] == str(old) for row in saved["slurm_batches"])
    assert validate_handoff(published[0])["status"] == "passed"
    result = control.repair_dft_handoff("decision", digest, 2)
    assert result["status"] == "prepared" and published[0] == published[1]
    saved = read_json(path)
    assert saved["other_process_update"] is True and len(saved["dft_handoff_repairs"]) == 1
    assert all(row["handoff"]["folder"] == str(published[0]) for row in saved["slurm_batches"])
    assert control.repair_dft_handoff("decision", digest, 2)["status"] == "reused"
    assert len(read_json(path)["dft_handoff_repairs"]) == 1


def test_existing_repair_receipt_rejects_concurrent_change(tmp_path, monkeypatch):
    import phase_agent.tools.remote.dft_handoff_repair as module
    import phase_agent.tools.state.execution_receipts as receipts

    root, state, _, digest = fixture(tmp_path)
    path = root / "workflow_state" / "state.json"
    path.parent.mkdir()
    write_json(path, state)
    monkeypatch.setattr(receipts, "recovery_report", lambda *_: {"unsettled": []})
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    control.repair_dft_handoff("decision", digest, 2)
    original = module.repair_dft_handoff

    def competing(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result["status"] == "reused"
        script = (
            "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
            "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
        )
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)
        return result

    monkeypatch.setattr(module, "repair_dft_handoff", competing)
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.repair_dft_handoff("decision", digest, 2)
    saved = read_json(path)
    assert saved["other_process_update"] is True and len(saved["dft_handoff_repairs"]) == 1
