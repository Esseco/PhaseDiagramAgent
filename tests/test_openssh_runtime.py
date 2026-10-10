import subprocess
from pathlib import Path

from phase_agent.tools.remote.openssh_runtime import OpenSSHJobScheduler


class FakeRunner:
    def __init__(self, outputs): self.outputs, self.calls = list(outputs), []
    def __call__(self, argv, **kwargs):
        self.calls.append(argv); code, output = self.outputs.pop(0)
        return subprocess.CompletedProcess(argv, code, stdout=output, stderr="")


def test_windows_path_arguments_work_with_configured_ssh_scripts():
    runner = FakeRunner([(0, "12345\n"), (0, "running\n"), (0, "cancelled\n")])
    scheduler = OpenSSHJobScheduler("BJ-HPC", submit_script="/data/home/user/bin/submit.sh",
        query_script="/data/home/user/bin/query.sh", cancel_script="/data/home/user/bin/cancel.sh",
        runner=runner)
    assert scheduler.submit(script_path=Path("C:/temp/task/submit.sbatch"), submission_token="id1")["job_id"] == "12345"
    assert scheduler.query(job_id=12345)["status"] == "running"
    assert scheduler.cancel(job_id=12345)["status"] == "cancelled"
    assert all(row[:2] == ["ssh", "BJ-HPC"] for row in runner.calls)
