"""Configurable scheduler boundary; no site command is guessed."""

from __future__ import annotations

import subprocess


class UnconfiguredSchedulerAdapter:
    def submit(self, script_path):
        return {"status": "not_configured", "reason": "submit command is not configured"}

    def query(self, job_id):
        return {"status": "not_configured", "job_id": job_id}

    def cancel(self, job_id):
        return {"status": "not_configured", "job_id": job_id}


class CommandSchedulerAdapter:
    def __init__(self, *, submit_command, query_command=None, cancel_command=None, runner=subprocess.run):
        if not submit_command:
            raise ValueError("submit_command is required")
        self.submit_command = list(submit_command)
        self.query_command = list(query_command or [])
        self.cancel_command = list(cancel_command or [])
        self.runner = runner

    def submit(self, script_path):
        completed = self.runner(
            [*self.submit_command, str(script_path)], check=True, capture_output=True, text=True,
        )
        words = completed.stdout.strip().split()
        job_id = words[-1] if words and words[-1].isdigit() else None
        if not job_id:
            return {"status": "submission_unverified", "stdout": completed.stdout.strip()}
        return {"status": "submitted", "job_id": job_id, "stdout": completed.stdout.strip()}

    def query(self, job_id):
        if not self.query_command:
            return {"status": "not_configured", "job_id": job_id}
        completed = self.runner(
            [*self.query_command, str(job_id)], check=True, capture_output=True, text=True,
        )
        return {"status": "queried", "job_id": job_id, "stdout": completed.stdout.strip()}

    def cancel(self, job_id):
        if not self.cancel_command:
            return {"status": "not_configured", "job_id": job_id}
        completed = self.runner(
            [*self.cancel_command, str(job_id)], check=True, capture_output=True, text=True,
        )
        return {"status": "cancelled", "job_id": job_id, "stdout": completed.stdout.strip()}


class MockSchedulerAdapter:
    def __init__(self):
        self.submissions = []

    def submit(self, script_path):
        job_id = f"mock-{len(self.submissions) + 1:06d}"
        self.submissions.append((str(script_path), job_id))
        return {"status": "submitted", "job_id": job_id}

    def query(self, job_id):
        return {"status": "completed", "job_id": job_id}

    def cancel(self, job_id):
        return {"status": "cancelled", "job_id": job_id}
