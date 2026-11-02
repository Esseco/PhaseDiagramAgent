"""Freeze automatic derived handoffs inside the original training authorization."""

from copy import deepcopy
from pathlib import Path
from phase_agent.tools.step_runner.file_protocol import content_id


def config_binding(config):
    settings = deepcopy(config.get("remote_training_validation") or {})
    data = Path(settings.get("data_path") or "__missing_validation_data__")
    if data.is_file():
        from phase_agent.tools.remote.transfer_bundle import checksum

        settings["data_sha256"] = checksum(data)
    template = config.get("remote_training_job_template")
    if template and Path(template).is_file():
        from phase_agent.tools.remote.transfer_bundle import checksum

        settings["template_sha256"] = checksum(template)
    return content_id(
        {
            "validation": settings,
            "template": template,
            "environments": config.get("python_environments"),
            "supercomputer": config.get("supercomputer"),
        },
        "training-handoff",
    )


def proposed_scope(action, config):
    kinds = (action.get("parameters") or {}).get("handoff_scope", ["manifest"])
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(kind not in {"manifest", "validation"} for kind in kinds)
    ):
        raise ValueError("handoff_scope must list manifest and/or validation")
    return {
        "schema": 1,
        "kinds": sorted(set(kinds)),
        "config_binding": config_binding(config),
        "submission_approved": False,
    }


def bind_job(scope, key, job):
    return {
        **deepcopy(scope),
        "job_key": key,
        "directory": job["directory"],
        "report_hash": content_id(job.get("report") or {}, "training-report"),
    }


def check_handoff(job, config, kind):
    authorization = job.get("handoff_authorization")
    # Pre-upgrade jobs retain their historical contract, not a newly invented approval.
    if authorization is None:
        return
    if kind not in authorization.get("kinds", []):
        raise ValueError("training_handoff_requires_new_approval:" + kind)
    if (
        authorization.get("directory") != job.get("directory")
        or authorization.get("report_hash")
        != content_id(job.get("report") or {}, "training-report")
        or authorization.get("config_binding") != config_binding(config)
    ):
        raise ValueError("training_handoff_authorization_changed; request a new proposal")
