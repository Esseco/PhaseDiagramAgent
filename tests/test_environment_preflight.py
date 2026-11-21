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


def test_dispatch_resumes_missing_input_and_skips_completed_jobs(tmp_path, monkeypatch):
    import zipfile
    from phase_agent.tools.remote import submission_dispatcher as dispatch
    from phase_agent.tools.remote.submission_handoff import prepare_handoff

    operation = tmp_path / "operation"
    job = operation / "inputs/job.sh"
    job.parent.mkdir(parents=True)
    job.write_text("#!/bin/bash\ntrue\n", newline="\n")
    (job.parent / "a-lookup.dat").write_text("validated before the interrupted member")
    root = Path(prepare_handoff(operation, [job])["folder"])
    monkeypatch.setenv("SLURM_JOB_ID", "local-test")
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(dispatch.shutil, "which", lambda _: "fake-sbatch")
    monkeypatch.setattr(dispatch, "preflight_environment", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "preflight_models", lambda *a: None)
    submissions = []

    def submit(command, **kwargs):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout="701\n", stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    original_match = dispatch._input_matches

    def interrupt(source, row, output=None):
        if output is not None and row["path"].endswith("/job.sh"):
            output.write(b"incomplete")
            raise OSError("interrupted extraction")
        return original_match(source, row, output)

    monkeypatch.setattr(dispatch, "_input_matches", interrupt)
    with pytest.raises(OSError, match="interrupted extraction"):
        dispatch.run(root)
    assert (root / "inputs/a-lookup.dat").exists()
    assert not (root / "inputs/job.sh").exists()
    assert not list(root.rglob("*.part"))
    assert not (root / ".submitted_jobs.json").exists()
    assert not submissions

    monkeypatch.setattr(dispatch, "_input_matches", original_match)
    names = []
    original_open = zipfile.ZipFile.open

    def track(archive, member, *args, **kwargs):
        names.append(member if isinstance(member, str) else member.filename)
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", track)
    (root / "inputs/a-lookup.dat").write_text("changed after validation")
    with pytest.raises(ValueError, match="Existing input changed"):
        dispatch.run(root)
    assert not submissions
    (root / "inputs/a-lookup.dat").write_text("validated before the interrupted member")

    names.clear()
    dispatch.run(root)
    assert names[:2] == ["transfer_manifest.json", "submission_plan.json"]
    assert names.count("inputs/job.sh") == 1
    assert "inputs/a-lookup.dat" not in names
    assert len(submissions) == 1

    names.clear()
    dispatch.run(root)
    assert names == ["transfer_manifest.json", "submission_plan.json"]
    assert len(submissions) == 1

    receipt_path = root / ".submitted_jobs.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["jobs"]["inputs/job.sh"] = {"status": "submitting"}
    receipt_path.write_text(json.dumps(receipt))
    names.clear()
    with pytest.raises(RuntimeError, match="check squeue/sacct"):
        dispatch.run(root)
    assert not any(name.endswith(("/a-lookup.dat", "/job.sh")) for name in names)
    assert len(submissions) == 1


@pytest.mark.parametrize("ambiguous", [False, True])
def test_qos_rejection_resumes_only_unsubmitted_child(tmp_path, monkeypatch, ambiguous):
    import zipfile

    from phase_agent.tools.remote import submission_dispatcher as dispatch
    from phase_agent.tools.remote.submission_handoff import prepare_handoff

    operation = tmp_path / "operation"
    inputs = operation / "inputs"
    inputs.mkdir(parents=True)
    jobs = [inputs / name for name in ("a.sh", "b.sh")]
    for job in jobs:
        job.write_text("#!/bin/bash\ntrue\n", newline="\n")
    root = Path(prepare_handoff(operation, jobs)["folder"])
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setattr(dispatch.shutil, "which", lambda _: "fake-sbatch")
    monkeypatch.setattr(dispatch, "preflight_environment", lambda *a, **k: None)
    attempted = []

    def sbatch(command, **kwargs):
        attempted.append(command[-1])
        if len(attempted) == 2:
            return SimpleNamespace(
                returncode=1,
                stdout="901\n" if ambiguous else "",
                stderr=(
                    "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
                    "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy"
                ),
            )
        return SimpleNamespace(returncode=0, stdout=f"{800 + len(attempted)}\n", stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", sbatch)
    with pytest.raises(RuntimeError, match="QOSMaxSubmitJobPerUserLimit"):
        dispatch.run(root)
    receipt = root / ".submitted_jobs.json"
    assert json.loads(receipt.read_text())["jobs"]["inputs/a.sh"]["job_id"] == "801"
    if ambiguous:
        assert json.loads(receipt.read_text())["jobs"]["inputs/b.sh"]["status"] == "submitting"
        with pytest.raises(RuntimeError, match="check squeue/sacct"):
            dispatch.run(root)
        assert attempted == ["a.sh", "b.sh"]
        return

    assert json.loads(receipt.read_text())["jobs"]["inputs/b.sh"]["status"] == "rejected"
    with zipfile.ZipFile(root.with_name(root.name[:-6] + "_output") / "diagnostics.zip") as archive:
        assert json.loads(archive.read("dispatch-error.json"))["code"] == "slurm_qos_submit_limit"
    dispatch.run(root)
    dispatch.run(root)
    assert attempted == ["a.sh", "b.sh", "b.sh"]
    assert json.loads(receipt.read_text())["jobs"]["inputs/b.sh"]["job_id"] == "803"
