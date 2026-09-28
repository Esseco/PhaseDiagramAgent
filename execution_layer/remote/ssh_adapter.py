"""Windows/Linux OpenSSH and SCP adapters for the remote file protocol.

The adapter invokes only caller-configured scripts. It does not assume a
particular scheduler or persist credentials. SSH keys remain in the user's
OpenSSH configuration/agent.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

from execution_layer.remote.transport import CommandTransferAdapter
from execution_layer.remote.scheduler import RemoteSchedulerAdapter


class OpenSSHTransport:
    """Transfer batch inputs and selected result files through system scp/ssh."""

    RESULT_FILE_NAMES = {
        "result.json", "task.finished.json", "final.vasp", "CONTCAR",
        "checkpoint.json", "checkpoint.json.gz", "log_index.json",
        "status_summary.json", "initial_relaxed.vasp",
    }

    def __init__(self, host: str, *, ssh="ssh", scp="scp", runner: Callable = subprocess.run):
        if not host or any(char.isspace() for char in host):
            raise ValueError("host must be one OpenSSH alias")
        self.host, self.ssh, self.scp, self.runner = host, ssh, scp, runner

    def upload_tree(self, local_path, remote_path):
        source = Path(local_path).resolve()
        remote = _remote_path(remote_path)
        snapshot_path = source / "batch.snapshot.json"
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        checksum = snapshot.get("manifest_checksum")
        if not checksum:
            raise ValueError("batch snapshot is missing manifest_checksum")
        parent, name = str(remote.parent), remote.name
        final = str(remote)
        existing = self._remote_snapshot(final)
        if existing is not None:
            if existing.get("manifest_checksum") != checksum:
                raise ValueError(f"remote batch already exists with different checksum: {final}")
            return {"status": "already_synced", "path": final, "checksum": checksum}

        stage_name = f".{name}.uploading-{checksum[:16]}"
        staging = str(PurePosixPath(parent) / stage_name)
        staged = self._remote_snapshot(staging)
        if staged is None:
            # Refuse to overwrite an incomplete prior transfer. It can be inspected
            # or removed manually after confirming no process is uploading it.
            exists = self._ssh("test", "-e", staging, check=False).returncode == 0
            if exists:
                raise RuntimeError(f"incomplete remote upload exists: {staging}")
            self._ssh("mkdir", "-p", parent)
            self._run([self.scp, "-r", str(source), f"{self.host}:{staging}"])
            staged = self._remote_snapshot(staging)
        if staged is None or staged.get("manifest_checksum") != checksum:
            raise ValueError("uploaded batch snapshot checksum does not match local batch")
        # mv is atomic on the remote filesystem when source and destination share a parent.
        self._ssh("mv", "--", staging, final)
        return {"status": "synced", "path": final, "checksum": checksum}

    def download_tree(self, remote_path, local_path):
        remote = _remote_path(remote_path)
        target = Path(local_path)
        target.mkdir(parents=True, exist_ok=True)
        manifest_text = self._ssh("cat", str(PurePosixPath(remote) / "manifest.json"), check=False)
        if manifest_text.returncode != 0:
            return {"status": "not_found", "path": remote}
        try:
            manifest = json.loads(manifest_text.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid remote manifest for {remote}") from error
        copied = []
        for row in manifest:
            task_dir = PurePosixPath(row["result_path"]).parent
            for filename in sorted(self.RESULT_FILE_NAMES):
                remote_file = PurePosixPath(remote) / task_dir / filename
                check = self._ssh("test", "-f", str(remote_file), check=False)
                if check.returncode != 0:
                    continue
                local_file = target / task_dir / filename
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self._run([self.scp, f"{self.host}:{remote_file}", str(local_file)])
                copied.append(str(task_dir / filename))
            # The scientific result names its exact final structure. Only copy
            # a relative path underneath this task directory, never arbitrary
            # paths supplied by a remote JSON file.
            remote_result = PurePosixPath(remote) / task_dir / "result.json"
            if self._ssh("test", "-f", str(remote_result), check=False).returncode == 0:
                result_text = self._ssh("cat", str(remote_result), check=False)
                if result_text.returncode == 0:
                    result = json.loads(result_text.stdout)
                    final = ((result.get("outputs") or {}).get("structure_path")
                             or (result.get("outputs") or {}).get("final_structure_path"))
                    if final:
                        from execution_layer.remote.resolve_final_structure import resolve_final_structure
                        relative = resolve_final_structure(final, str(PurePosixPath(remote) / task_dir))
                        if relative:
                            source = PurePosixPath(remote) / task_dir / relative
                            if self._ssh("test", "-f", str(source), check=False).returncode == 0:
                                destination = target / task_dir / relative
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                self._run([self.scp, f"{self.host}:{source}", str(destination)])
                                copied.append(str(task_dir / relative))
        # Small batch status and log indexes are useful even if no task completed.
        for filename in ("job-status.json", "status_summary.json", "log_index.json"):
            remote_file = PurePosixPath(remote) / filename
            if self._ssh("test", "-f", str(remote_file), check=False).returncode == 0:
                self._run([self.scp, f"{self.host}:{remote_file}", str(target / filename)])
                copied.append(filename)
        return {"status": "synced", "path": str(target), "files": copied}

    def _remote_snapshot(self, remote_directory):
        path = str(PurePosixPath(remote_directory) / "batch.snapshot.json")
        result = self._ssh("cat", path, check=False)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid remote batch snapshot: {path}") from error

    def _ssh(self, *remote_argv, check=True):
        # Pass an explicitly quoted remote command; OpenSSH executes it through a shell.
        command = " ".join(shlex.quote(str(arg)) for arg in remote_argv)
        return self._run([self.ssh, self.host, command], check=check, capture_output=True, text=True)

    def _run(self, argv, *, check=True, capture_output=True, text=True):
        return self.runner(argv, check=check, capture_output=capture_output, text=text)


class OpenSSHJobScheduler:
    """Call configured remote submit/query/cancel scripts over SSH.

    ``find_submission_command`` is optional. When a submit response is lost,
    callers must configure a token-aware lookup before an uncertain submission
    can be retried safely.
    """

    def __init__(self, host: str, *, submit_script: str, query_script: str,
                 cancel_script: str, find_submission_command=None,
                 ssh="ssh", runner: Callable = subprocess.run):
        self.host, self.ssh, self.runner = host, ssh, runner
        self.submit_script = _remote_path(submit_script)
        self.query_script = _remote_path(query_script)
        self.cancel_script = _remote_path(cancel_script)
        self.find_submission_command = find_submission_command

    def submit(self, *, script_path, submission_token):
        output = self._call(self.submit_script, script_path)
        job_id = _job_id(output)
        if not job_id:
            return {"status": "submission_uncertain", "submission_token": submission_token,
                    "stdout": output.strip()}
        return {"status": "submitted", "job_id": job_id, "submission_token": submission_token,
                "stdout": output.strip()}

    def query(self, *, job_id=None, submission_token=None):
        if job_id is None:
            if not self.find_submission_command:
                return {"status": "not_configured", "submission_token": submission_token}
            output = self._call(self.find_submission_command, submission_token)
            try:
                record = json.loads(output)
            except json.JSONDecodeError:
                return {"status": "unknown", "submission_token": submission_token, "stdout": output.strip()}
            return record
        output = self._call(self.query_script, str(job_id)).strip()
        if output not in {"submitted", "running", "completed", "failed", "cancelled", "not_found"}:
            return {"status": "unknown", "job_id": str(job_id), "stdout": output}
        return {"status": output, "job_id": str(job_id)}

    def cancel(self, *, job_id):
        output = self._call(self.cancel_script, str(job_id)).strip()
        return {"status": "cancelled" if output == "cancelled" else "unknown",
                "job_id": str(job_id), "stdout": output}

    def _call(self, script, argument):
        remote_command = " ".join(shlex.quote(value) for value in (script, str(argument)))
        result = self.runner([self.ssh, self.host, remote_command], check=True,
                             capture_output=True, text=True)
        return result.stdout


def create_openssh_adapters(host: str, *, remote_batch_root: str,
                            submit_script: str, query_script: str, cancel_script: str,
                            find_submission_command=None, ssh="ssh", scp="scp", runner=subprocess.run):
    """Return protocol-compatible transport and scheduler instances."""
    return {
        "transport": CommandTransferAdapter(
            upload=OpenSSHTransport(host, ssh=ssh, scp=scp, runner=runner).upload_tree,
            download=OpenSSHTransport(host, ssh=ssh, scp=scp, runner=runner).download_tree,
        ),
        "scheduler": OpenSSHJobScheduler(
            host, submit_script=submit_script, query_script=query_script,
            cancel_script=cancel_script, find_submission_command=find_submission_command,
            ssh=ssh, runner=runner,
        ),
        "remote_batch_root": str(_remote_path(remote_batch_root)),
    }


def _remote_path(value):
    path = PurePosixPath(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"remote path must be absolute and must not contain '..': {value}")
    return path


def _job_id(output):
    text = str(output).strip()
    token = text.splitlines()[-1].split(";", 1)[0].strip() if text else ""
    return token if token.isdigit() else None
