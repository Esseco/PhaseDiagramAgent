"""Prepare one explicitly approved training follow-up without submission."""

from copy import deepcopy
from phase_agent.tools.step_runner.file_protocol import content_id
from phase_agent.policies.training_handoff_scope import bind_job


def request_binding(action, state, config):
    from phase_agent.policies.training_handoff_scope import proposed_scope

    params = action.get("parameters") or {}
    key, kind = params.get("job_key"), params.get("kind")
    job = (state.get("remote_finetune_jobs") or {}).get(key)
    if not job or kind not in {"manifest", "validation"}:
        raise ValueError(
            "training follow-up requires an existing job_key and manifest/validation kind"
        )
    scope = proposed_scope({"parameters": {"handoff_scope": [kind]}}, config)
    return {
        "job_key": key,
        "kind": kind,
        "scope": bind_job(scope, key, job),
        "job_hash": content_id(
            {
                "report": job.get("report"),
                "directory": job["directory"],
                "original_model_version": job.get("original_model_version"),
            },
            "handoff-job",
        ),
    }


def prepare_training_followup(*, action, context):
    from phase_agent.tools.local.training_handoff import (
        prepare_manifest_job,
        prepare_validation_job,
        validation_request,
        record_handoff,
    )
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    state = deepcopy(context.get("event_state") or {})
    config = context.get("effective_config") or {}
    binding = request_binding(action, state, config)
    if binding != action.get("_training_handoff_request"):
        raise ValueError("training follow-up changed since proposal; request new approval")
    key = binding["job_key"]
    job = state["remote_finetune_jobs"][key]
    job["handoff_authorization"] = binding["scope"]
    result = inspect_training_results(job)
    if not result or result.get("status") == "invalid_results":
        raise ValueError("training result evidence unavailable for follow-up")
    handoff = {"training_fingerprint": result.get("fingerprint"), "submitted": False}
    if binding["kind"] == "manifest":
        handoff = prepare_manifest_job(job, config, handoff)
    else:
        request = validation_request(state, key, job, config, result)
        if request.get("missing"):
            raise ValueError("validation prerequisites missing: " + "; ".join(request["missing"]))
        handoff = prepare_validation_job(job, config, request, handoff)
    state = record_handoff(state, key, handoff)
    return {
        "status": "prepared",
        "state": state,
        "handoffs": [handoff],
        "submitted": False,
        "actual_cost": 0.0,
    }
