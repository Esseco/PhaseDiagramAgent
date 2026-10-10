import json
import subprocess
from pathlib import Path

import pytest

from phase_agent.tools.remote.ssh_adapter import OpenSSHJobScheduler, OpenSSHTransport


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return subprocess.CompletedProcess(argv, output[0], stdout=output[1], stderr="")


def test_query_submit_and_cancel_call_only_configured_remote_scripts():
    runner = FakeRunner([(0, "12345\n"), (0, "running\n"), (0, "cancelled\n")])
    adapter = OpenSSHJobScheduler(
        "BJ-HPC", submit_script="/data/home/user/bin/submit.sh",
        query_script="/data/home/user/bin/query.sh", cancel_script="/data/home/user/bin/cancel.sh",
        runner=runner,
    )
    assert adapter.submit(script_path="/data/home/user/batches/b1/submit.sbatch",
                          submission_token="token-1")["job_id"] == "12345"
    assert adapter.query(job_id="12345")["status"] == "running"
    assert adapter.cancel(job_id="12345")["status"] == "cancelled"
    assert len(runner.calls) == 3
    assert all(call[0][0:2] == ["ssh", "BJ-HPC"] for call in runner.calls)


def test_unknown_submit_output_never_becomes_a_job_id():
    adapter = OpenSSHJobScheduler(
        "BJ-HPC", submit_script="/bin/submit", query_script="/bin/query", cancel_script="/bin/cancel",
        runner=FakeRunner([(0, "submission accepted, id pending\n")]),
    )
    result = adapter.submit(script_path="/remote/job.sbatch", submission_token="t1")
    assert result["status"] == "submission_uncertain"
    assert "job_id" not in result


def test_uncertain_submit_requires_token_lookup_before_retry():
    adapter = OpenSSHJobScheduler(
        "BJ-HPC", submit_script="/bin/submit", query_script="/bin/query", cancel_script="/bin/cancel",
        runner=FakeRunner([]),
    )
    assert adapter.query(job_id=None, submission_token="t1")["status"] == "not_configured"


def test_download_copies_only_whitelisted_files_and_never_trajectories(tmp_path):
    manifest = [{"result_path": "00000-T1/result.json"}]
    class DownloadRunner:
        def __init__(self): self.calls = []
        def __call__(self, argv, **kwargs):
            self.calls.append(argv)
            if argv[0] == "ssh" and argv[-1].startswith("cat "):
                body = manifest if argv[-1].endswith("manifest.json") else {}
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(body), stderr="")
            if argv[0] == "scp" and "task.finished.json" in argv[-1]:
                Path(argv[-1]).write_text("{}")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    runner = DownloadRunner(); transport = OpenSSHTransport("BJ-HPC", runner=runner)
    result = transport.download_tree("/remote/batches/b1", tmp_path)
    assert result["status"] == "synced"
    scp_targets = [row[-1] for row in runner.calls if row[0] == "scp"]
    assert any("task.finished.json" in value for value in scp_targets)
    assert not any("traj" in value for value in scp_targets)


@pytest.mark.parametrize("value", ["relative/path", "/a/../b"])
def test_remote_paths_must_be_absolute_and_canonical(value):
    with pytest.raises(ValueError):
        OpenSSHTransport("BJ-HPC").upload_tree(".", value)
