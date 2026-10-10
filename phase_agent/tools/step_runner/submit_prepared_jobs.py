"""Submit prepared jobs once from a network-capable login node."""

from copy import deepcopy

from phase_agent.tools.step_runner.check_node_role import check_node_role
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def submit_prepared_jobs(state_path, *, scheduler, node_role=None, limit=None):
    check_node_role("login", role=node_role)
    state = read_json(state_path, {}) or {}
    submitted = []
    batches = state.setdefault("slurm_batches", [])
    for batch in batches:
        if batch.get("status") in {"submitting", "submission_uncertain"}:
            found = _query_uncertain(scheduler, batch)
            if found.get("job_id"):
                batch.update(status=found.get("status", "submitted"), job_id=found["job_id"])
                write_json(state_path, state)
            else:
                batch["status"] = "submission_uncertain"
                write_json(state_path, state)
                continue
        if batch.get("status") != "prepared" or batch.get("job_id"):
            continue
        if limit is not None and len(submitted) >= int(limit):
            break
        # Persist intent before contacting the scheduler.  A disconnect after
        # scheduler acceptance therefore becomes an explicit uncertain state,
        # never an automatic duplicate submission.
        batch["status"] = "submitting"
        batch["submission_attempt_id"] = f"submit-{batch['batch_id']}"
        write_json(state_path, state)
        result = scheduler.submit(batch["script_path"])
        if result.get("status") != "submitted" or not result.get("job_id"):
            if result.get("status") == "not_configured":
                batch["status"] = "prepared"
                write_json(state_path, state)
                return {"status": "not_configured", "state": state, "submitted": submitted}
            batch["status"] = "submission_uncertain"
            write_json(state_path, state)
            raise RuntimeError(f"scheduler did not confirm submission: {result}")
        batch.update({"status": "submitted", "job_id": result["job_id"]})
        for task in state.get("tasks") or []:
            if task.get("slurm_batch_id") == batch.get("batch_id"):
                task.update({"job_id": result["job_id"], "status": "pending"})
        submitted.append({"batch_id": batch["batch_id"], "job_id": result["job_id"]})
        write_json(state_path, state)
    write_json(state_path, state)
    return {
        "status": "submitted" if submitted else "nothing_to_submit",
        "state": deepcopy(state),
        "submitted": submitted,
    }


def _query_uncertain(scheduler, batch):
    """Never retry an ambiguous submit without first resolving its identity."""
    token = batch.get("submission_attempt_id") or f"submit-{batch['batch_id']}"
    try:
        return scheduler.query(job_id=batch.get("job_id"), submission_token=token)
    except TypeError:
        if batch.get("job_id") is None:
            return {"status": "not_configured", "reason": "submission_token_lookup_not_supported"}
        return scheduler.query(batch["job_id"])
