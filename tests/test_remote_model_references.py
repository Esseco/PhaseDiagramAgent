import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.science.mlip.slurm_executor import remote_model_references
from phase_agent.tools.remote.submission_dispatcher import run
from phase_agent.tools.remote.submission_handoff import prepare_handoff


def test_remote_only_committee_never_probes_local_weights(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda path: pytest.fail("local file probe"))
    model = {
        "version": "v1",
        "members": [
            {"remote_model_path": f"/hpc/com_{i}.model", "sha256": str(i)} for i in range(4)
        ],
    }
    main, paths, evidence = remote_model_references(model)
    assert main is None and len(paths) == 4
    assert [x["sha256"] for x in evidence] == ["0", "1", "2", "3"]
    assert remote_model_references({"remote_model_path": "/hpc/base.model"})[0] == "/hpc/base.model"


@pytest.mark.parametrize(
    "model",
    [
        {},
        {"remote_model_paths": [""]},
        {"model_paths": ["a", "a"]},
        {"model_paths": ["a"], "members": [{}, {}]},
        {"model_paths": ["a"], "main_model_index": 2},
    ],
)
def test_invalid_references_rejected_without_loading(model):
    with pytest.raises(ValueError, match="remote_model_reference"):
        remote_model_references(model)


def test_remote_preflight_failure_returns_cause_without_submitting(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    job = package / "job"
    job.mkdir(parents=True)
    script = job / "GPU.sh"
    script.write_text("#!/bin/bash\n#SBATCH --gres=gpu:1\ntrue\n")
    (job / "task.json").write_text(
        json.dumps(
            {
                "stage": "relax_and_feature",
                "worker_job": {"model_path": str(tmp_path / "missing.model")},
            }
        )
    )
    handoff = prepare_handoff(package, [str(script)])
    folder = Path(handoff["folder"])
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setattr("shutil.which", lambda name: "fake-sbatch")
    monkeypatch.setitem(
        sys.modules, "fcntl", SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2)
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("No submission allowed"))
    with pytest.raises(FileNotFoundError, match="remote_model_not_found"):
        run(folder)
    error = json.loads((folder / "return-results/dispatch-error.json").read_text())
    assert (
        error["location"] == "remote" and "local model download is not required" in error["remedy"]
    )
    receipt = json.loads((folder / ".submitted_jobs.json").read_text())
    assert receipt["jobs"] == {}
    assert receipt["model_preflight"]["status"] == "failed"
