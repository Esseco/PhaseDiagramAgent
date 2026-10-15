from pathlib import Path
import pytest
import importlib.util
import phase_agent

_helpers_path = Path(phase_agent.__file__).parents[1] / "tests/test_relax_mc_batch.py"
_spec = importlib.util.spec_from_file_location("relax_mc_dependency_helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
package, fake_executor = _helpers.package, _helpers.fake_executor
from phase_agent.tools.relax_mc_batch.run import SlurmIO
from phase_agent.tools.relax_mc_batch.dependencies import advance
from phase_agent.tools.relax_mc_batch.files import read, rebase, coordinator_profile


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("d")


def run_batch(root, *, submit, io, phase=None):
    # Exercise controllers directly; immutable inventory validation is covered
    # separately by the existing run_batch corruption regression.
    assert submit
    return advance(root, rebase(read(root / "batch.json"), root), io, phase=phase)


class FakeSlurm:
    def __init__(self, root, execute, manager):
        self.root, self.execute, self.manager = root, execute, manager
        self.rows = []
        self.running = set()
        self.missing = False
        self.uncertain = False

    def submit(self, path, *, dependency=None):
        job = str(100 + len(self.rows))
        self.rows.append((job, str(path), dependency))
        if self.uncertain:
            raise RuntimeError("lost receipt")
        if dependency is None and not self.missing:
            state = rebase(read(self.root / "workflow_state.json"), self.root)
            batch = next(b for b in state["slurm_batches"] if b["script_path"] == str(path))
            self.execute(batch, state, self.manager)
        return job

    def active(self, ids):
        return set(ids) & self.running


def setup(tmp_path, monkeypatch, **kwargs):
    manager, _, _, _, _, result = package(tmp_path)
    execute, calls = fake_executor(monkeypatch, **kwargs)
    root = Path(result["directory"])
    return root, FakeSlurm(root, execute, manager), calls


def test_parallel_phase_dependencies_and_complete_feedback(tmp_path, monkeypatch):
    root, io, calls = setup(tmp_path, monkeypatch)
    first = run_batch(root, submit=True, io=io)
    assert first["status"] == "queued"
    assert len(first["gpu_job_ids"]) == 3
    assert io.rows[-1][2] == "afterany:" + ":".join(first["gpu_job_ids"])
    assert "--partition=p1" in Path(io.rows[-1][1]).read_text()
    assert "--gres" not in Path(io.rows[-1][1]).read_text()
    assert "--partition=p1" in (root / "submit_all.sh").read_text()
    count = len(io.rows)
    assert run_batch(root, submit=True, io=io)["status"] == "queued"
    assert len(io.rows) == count
    for phase in range(4):
        outcome = run_batch(root, submit=True, io=io, phase=phase)
        if phase < 3:
            assert outcome["status"] == "queued", outcome
            assert io.rows[-1][2] == "afterany:" + ":".join(outcome["gpu_job_ids"])
            count = len(io.rows)
            assert run_batch(root, submit=True, io=io, phase=phase)["status"] == "stale_phase"
            assert len(io.rows) == count
        else:
            assert outcome["status"] == "completed", outcome
    assert len(calls) == 49
    assert len(io.rows) == 16  # twelve GPU jobs and four short analysis jobs
    assert read(root / "workflow_state.json")["relax_mc_remote"]["completed_mc_waves"] == [0, 1, 2]
    assert run_batch(root, submit=True, io=io)["submitted"] is False


@pytest.mark.parametrize("missing", [False, True])
def test_failed_or_missing_relax_blocks_mc(tmp_path, monkeypatch, missing):
    root, io, _ = setup(tmp_path, monkeypatch, fail_relax=True)
    io.missing = missing
    run_batch(root, submit=True, io=io)
    count = len(io.rows)
    outcome = run_batch(root, submit=True, io=io, phase=0)
    assert outcome["status"] == "paused", outcome
    assert len(io.rows) == count
    assert (root / "results.zip").is_file()


def test_uncertain_receipt_never_repeats_submission(tmp_path, monkeypatch):
    root, io, _ = setup(tmp_path, monkeypatch)
    io.uncertain = True
    assert run_batch(root, submit=True, io=io)["status"] == "failed"
    state = read(root / "workflow_state.json")
    assert state["slurm_batches"][0]["submission_state"] == "submitting"
    assert not state["slurm_batches"][0].get("job_id")
    assert run_batch(root, submit=True, io=io)["submitted"] is False
    assert len(io.rows) == 1


def test_sbatch_dependency_receipt(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import subprocess

    calls = []

    def capture(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="234;cluster\n")

    monkeypatch.setattr(subprocess, "run", capture)
    assert SlurmIO().submit(tmp_path / "job.sh", dependency="afterany:1:2") == "234"
    assert calls[0][2] == "--dependency=afterany:1:2"


def test_cpu_profile_rejects_gpu():
    with pytest.raises(ValueError, match="must_not_request_GPU"):
        coordinator_profile(
            {"relax_mc_cpu": {"slurm_options": {"partition": "p1", "gres": "gpu:1"}}}
        )


def test_terminal_analysis_publishes_named_output(tmp_path):
    from phase_agent.tools.relax_mc_batch.upload_driver import publish_output
    import json

    work = tmp_path / "epoch0_RelaxMC_Batch01_input" / "work"
    work.mkdir(parents=True)
    (work / "results.zip").write_bytes(b"verified archive")
    publish_output(work, {"status": "completed"})
    output = tmp_path / "epoch0_RelaxMC_Batch01_output"
    assert (output / "results.zip").read_bytes() == b"verified archive"
    assert json.loads((output / "run_status.json").read_text())["status"] == "completed"
