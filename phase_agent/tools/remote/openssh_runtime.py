"""Windows/Linux OpenSSH runtime wiring using the existing protocol APIs.

Use this module for the local OpenSSH alias (for example ``BJ-HPC``). It keeps
remote paths POSIX and passes every remote argument through shell quoting.
"""

from __future__ import annotations

import json
import shlex
import subprocess

from phase_agent.tools.remote.scheduler import RemoteSchedulerAdapter
from phase_agent.tools.remote.ssh_adapter import OpenSSHTransport, _job_id, _remote_path
from phase_agent.tools.remote.transport import CommandTransferAdapter


class OpenSSHJobScheduler(RemoteSchedulerAdapter):
    def __init__(
        self,
        host,
        *,
        submit_script,
        query_script,
        cancel_script,
        find_submission_command=None,
        ssh="ssh",
        runner=subprocess.run,
    ):
        self.host, self.ssh, self.runner = host, ssh, runner
        self.submit_script = str(_remote_path(submit_script))
        self.query_script = str(_remote_path(query_script))
        self.cancel_script = str(_remote_path(cancel_script))
        self.find_submission_command = (
            str(_remote_path(find_submission_command)) if find_submission_command else None
        )

    def submit(self, *, script_path, submission_token):
        output = self._call(self.submit_script, script_path)
        job_id = _job_id(output)
        if not job_id:
            return {
                "status": "submission_uncertain",
                "submission_token": submission_token,
                "stdout": output.strip(),
            }
        return {
            "status": "submitted",
            "job_id": job_id,
            "submission_token": submission_token,
            "stdout": output.strip(),
        }

    def query(self, *, job_id=None, submission_token=None):
        if job_id is None:
            if not self.find_submission_command:
                return {"status": "not_configured", "submission_token": submission_token}
            output = self._call(self.find_submission_command, submission_token)
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return {
                    "status": "unknown",
                    "submission_token": submission_token,
                    "stdout": output.strip(),
                }
        output = self._call(self.query_script, job_id).strip()
        if output not in {"submitted", "running", "completed", "failed", "cancelled", "not_found"}:
            return {"status": "unknown", "job_id": str(job_id), "stdout": output}
        return {"status": output, "job_id": str(job_id)}

    def cancel(self, *, job_id):
        output = self._call(self.cancel_script, job_id).strip()
        return {
            "status": "cancelled" if output == "cancelled" else "unknown",
            "job_id": str(job_id),
            "stdout": output,
        }

    def _call(self, script, argument):
        remote_command = " ".join(shlex.quote(str(value)) for value in (script, argument))
        result = self.runner(
            [self.ssh, self.host, remote_command], check=True, capture_output=True, text=True
        )
        return result.stdout


def create_openssh_adapters(
    host,
    *,
    remote_batch_root,
    submit_script,
    query_script,
    cancel_script,
    find_submission_command=None,
    ssh="ssh",
    scp="scp",
    runner=subprocess.run,
):
    transport = OpenSSHTransport(host, ssh=ssh, scp=scp, runner=runner)
    return {
        "transport": CommandTransferAdapter(
            upload=transport.upload_tree, download=transport.download_tree
        ),
        "scheduler": OpenSSHJobScheduler(
            host,
            submit_script=submit_script,
            query_script=query_script,
            cancel_script=cancel_script,
            find_submission_command=find_submission_command,
            ssh=ssh,
            runner=runner,
        ),
        "remote_batch_root": str(_remote_path(remote_batch_root)),
    }
