import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from phase_agent.tools.remote import environment_preflight as check


def test_module_check_does_not_import_large_libraries(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: checked.append(name) or object())
    monkeypatch.setattr(check.shutil, "which", lambda name: "/bin/bash")
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    before = set(sys.modules)
    result = check.preflight_environment(tmp_path, [script.name, script.name], required_modules=["torch", "torch"])
    assert checked == ["torch"] and set(sys.modules) == before
    assert result["scripts_checked"] == 1 and result["gpu_runtime"] == "not_checked"


def test_missing_dependency_has_recovery_without_submitting(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    with pytest.raises(check.EnvironmentPreflightError) as caught:
        check.preflight_environment(tmp_path, [], required_modules=["mace"])
    assert caught.value.report["missing"] == ["mace"]
    assert not caught.value.report["submitted"] and caught.value.report["next_step"]


def test_shell_check_is_parse_only_and_drops_startup_hooks(tmp_path, monkeypatch):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\nif broken\n")
    monkeypatch.setenv("BASH_ENV", "unsafe-init")
    monkeypatch.setattr(check.shutil, "which", lambda _: "/bin/bash")
    calls = []
    def syntax(command, **kwargs):
        calls.append(command)
        assert "BASH_ENV" not in kwargs["env"] and kwargs["timeout"] == 5
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(check.subprocess, "run", syntax)
    with pytest.raises(check.EnvironmentPreflightError) as caught:
        check.preflight_environment(tmp_path, [script.name])
    assert calls[0][1] == "-n" and caught.value.report["code"] == "shell_syntax"


def test_same_dispatch_entry_recovers_preflight_but_never_replays_submitted_jobs(tmp_path, monkeypatch):
    from phase_agent.tools.remote import submission_dispatcher as dispatch
    from phase_agent.tools.remote.submission_handoff import prepare_handoff

    operation = tmp_path / "operation"
    job = operation / "inputs/batch/job.sh"
    job.parent.mkdir(parents=True)
    job.write_text("#!/bin/bash\ntrue\n", newline="\n")
    handoff = prepare_handoff(operation, [job])
    root = Path(handoff["folder"])
    monkeypatch.setenv("SLURM_JOB_ID", "fake-only")
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(dispatch.shutil, "which", lambda _: "fake-sbatch")
    attempts, submissions = [], []
    def environment(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise check.EnvironmentPreflightError("missing_modules", missing=["mace"])
    monkeypatch.setattr(dispatch, "preflight_environment", environment)
    monkeypatch.setattr(dispatch, "preflight_models", lambda *args: 0)
    def submit(command, **kwargs):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    with pytest.raises(check.EnvironmentPreflightError):
        dispatch.run(root)
    assert not submissions
    receipt = root / ".submitted_jobs.json"
    assert json.loads(receipt.read_text())["jobs"] == {}
    dispatch.run(root)
    dispatch.run(root)
    assert len(submissions) == 1 and len(attempts) == 2
    assert json.loads(receipt.read_text())["jobs"]["inputs/batch/job.sh"]["job_id"] == "12345"
