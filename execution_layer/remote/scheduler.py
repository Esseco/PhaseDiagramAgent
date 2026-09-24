"""Remote login-node scheduler interface."""

class RemoteSchedulerAdapter:
    def __init__(self, *, submit, query, cancel=None, find_submission=None):
        self._submit, self._query = submit, query
        self._cancel, self._find_submission = cancel, find_submission

    def submit(self, *, script_path, submission_token):
        return self._submit(script_path=script_path, submission_token=submission_token)

    def query(self, *, job_id=None, submission_token=None):
        if job_id is None and self._find_submission is not None:
            return self._find_submission(submission_token=submission_token)
        return self._query(job_id=job_id, submission_token=submission_token)

    def cancel(self, *, job_id):
        if self._cancel is None:
            return {"status": "not_configured", "job_id": job_id}
        return self._cancel(job_id=job_id)


class MockRemoteScheduler:
    def __init__(self):
        self.jobs, self.submit_calls = {}, 0

    def submit(self, *, script_path, submission_token):
        if submission_token in self.jobs:
            return dict(self.jobs[submission_token])
        self.submit_calls += 1
        record = {"status": "submitted", "job_id": f"mock-{self.submit_calls:06d}",
                  "script_path": str(script_path), "submission_token": submission_token}
        self.jobs[submission_token] = record
        return dict(record)

    def query(self, *, job_id=None, submission_token=None):
        if submission_token in self.jobs:
            return dict(self.jobs[submission_token])
        match = next((row for row in self.jobs.values() if row["job_id"] == job_id), None)
        return dict(match) if match else {"status": "not_found", "job_id": job_id}

    def cancel(self, *, job_id):
        match = next((row for row in self.jobs.values() if row["job_id"] == job_id), None)
        if match:
            match["status"] = "cancelled"
            return dict(match)
        return {"status": "not_found", "job_id": job_id}
