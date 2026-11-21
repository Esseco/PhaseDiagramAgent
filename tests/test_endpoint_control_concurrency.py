"""Concurrent endpoint registration preserves immutable evidence for safe retry."""

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_endpoint_supplement_registration import case


def test_endpoint_conflict_preserves_evidence_and_can_be_revalidated(tmp_path, monkeypatch):
    import phase_agent.evidence.endpoint_supplement as module

    state, csvpath, archive, kwargs = case(tmp_path)
    state_path = tmp_path / "state.json"
    write_json(state_path, state)
    original = module.register_endpoint_supplement
    first = True
    published = []

    def competing(*args, **params):
        nonlocal first
        result = original(*args, **params)
        if result[1].get("directory"):
            published.append(Path(result[1]["directory"]))
        if first:
            first = False
            script = (
                "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
                "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
            )
            subprocess.run([sys.executable, "-c", script, str(state_path)], check=True)
        return result

    monkeypatch.setattr(module, "register_endpoint_supplement", competing)
    control = LocalAgentControl(SimpleNamespace(state_path=state_path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.register_endpoint_supplement(str(csvpath), str(archive), **kwargs)
    saved = read_json(state_path)
    assert saved["other_process_update"] is True and "endpoint_supplement" not in saved
    assert (published[0] / "evidence.csv").is_file() and (published[0] / "frames.zip").is_file()
    receipt = control.register_endpoint_supplement(str(csvpath), str(archive), **kwargs)
    saved = read_json(state_path)
    assert receipt["status"] == "registered" and published[0] == published[1]
    assert saved["other_process_update"] is True
    assert len(saved["tasks"][0]["outputs"]["extrapolation_failures"]) == 1
    assert module.supplement_facts(saved)["failure_count"] == 1
    assert (
        control.register_endpoint_supplement(str(csvpath), str(archive), **kwargs)["status"]
        == "already_registered"
    )


def test_existing_endpoint_receipt_rejects_concurrent_change(tmp_path, monkeypatch):
    import phase_agent.evidence.endpoint_supplement as module

    state, csvpath, archive, kwargs = case(tmp_path)
    state, _ = module.register_endpoint_supplement(state, csvpath, archive, **kwargs)
    path = tmp_path / "state.json"
    write_json(path, state)
    original = module.register_endpoint_supplement

    def competing(*args, **params):
        result = original(*args, **params)
        assert result[1]["status"] == "already_registered"
        script = (
            "import sys; from phase_agent.tools.step_runner.file_protocol import read_json, write_json; "
            "p=sys.argv[1]; s=read_json(p); s['other_process_update']=True; write_json(p,s)"
        )
        subprocess.run([sys.executable, "-c", script, str(path)], check=True)
        return result

    monkeypatch.setattr(module, "register_endpoint_supplement", competing)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match="stale_state_file_version"):
        control.register_endpoint_supplement(str(csvpath), str(archive), **kwargs)
    assert read_json(path)["other_process_update"] is True
