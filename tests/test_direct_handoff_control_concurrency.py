"""Named DFT packages remain recoverable after concurrent state updates."""

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_direct_dft_handoff_naming import setup


@pytest.mark.parametrize("tamper", ["none", "script", "remote", "extra", "output"])
def test_naming_conflict_preserves_and_revalidates_package(tmp_path, monkeypatch, tamper):
    import phase_agent.tools.state.execution_receipts as receipts
    import phase_agent.tools.remote.direct_dft_handoff_naming as module
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(receipts, "recovery_report", lambda *_: {"unsettled": []})
    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda _: True)
    root, state, old, digest = setup(tmp_path)
    path = root / "workflow_state" / "state.json"
    path.parent.mkdir()
    write_json(path, state)
    original = module.name_direct_dft_handoff
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

    monkeypatch.setattr(module, "name_direct_dft_handoff", competing)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.name_direct_dft_handoff("decision", digest, "/remote/project")
    saved = read_json(path)
    assert saved["other_process_update"] is True and not saved.get("handoff_naming_history")
    assert all(row["handoff"]["folder"] == str(old) for row in saved["slurm_batches"])
    assert (published[0] / "inputs.zip").is_file()
    if tamper != "none":
        remote = "/remote/project"
        if tamper == "script":
            (published[0] / "submit_all.sh").write_text("tampered", encoding="utf8")
        elif tamper == "remote":
            remote = "/remote/other"
        elif tamper == "extra":
            (published[0] / "unexpected.txt").write_text("preserve", encoding="utf8")
        elif tamper == "output":
            published[0].with_name(published[0].name[:-6] + "_output").mkdir()
        with pytest.raises(ValueError, match="already_exists|mismatch"):
            control.name_direct_dft_handoff("decision", digest, remote)
        assert published[0].is_dir()
        assert not read_json(path).get("handoff_naming_history")
        return
    result = control.name_direct_dft_handoff("decision", digest, "/remote/project")
    assert result["status"] == "prepared" and published[0] == published[1]
    saved = read_json(path)
    assert saved["other_process_update"] is True and len(saved["handoff_naming_history"]) == 1
    assert all(row["handoff"]["folder"] == str(published[0]) for row in saved["slurm_batches"])


def test_naming_save_failure_keeps_package_for_safe_retry(tmp_path, monkeypatch):
    import phase_agent.tools.state.execution_receipts as receipts
    import phase_agent.tools.step_runner.file_protocol as protocol
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(receipts, "recovery_report", lambda *_: {"unsettled": []})
    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda _: True)
    root, state, old, digest = setup(tmp_path)
    path = root / "workflow_state" / "state.json"
    path.parent.mkdir()
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    original = protocol.write_json_if_version

    def failed(*args, **kwargs):
        raise OSError("simulated write interruption")

    monkeypatch.setattr(protocol, "write_json_if_version", failed)
    with pytest.raises(OSError, match="simulated write interruption"):
        control.name_direct_dft_handoff("decision", digest, "/remote/project")
    folder = old.parent / "ExtrapolationRecovery_DFTSinglePoint_Batch01_input"
    assert (folder / "inputs.zip").is_file() and not read_json(path).get("handoff_naming_history")
    monkeypatch.setattr(protocol, "write_json_if_version", original)
    result = control.name_direct_dft_handoff("decision", digest, "/remote/project")
    assert result["handoff"]["folder"] == str(folder)
    assert len(read_json(path)["handoff_naming_history"]) == 1
