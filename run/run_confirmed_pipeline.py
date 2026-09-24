"""Internal deterministic pipeline gated by a confirmed configuration snapshot."""

from run.run_pipeline import run_pipeline


def run_confirmed_pipeline(manager, phase_references, run_config, config_session, **kwargs):
    if config_session.get("status") != "confirmed" or not config_session.get("confirmed_snapshot"):
        return {"status": "rejected", "reason": "configuration_not_confirmed", "submitted": False}
    result = run_pipeline(manager, phase_references, run_config, **kwargs)
    result["config_version"] = config_session["confirmed_snapshot"]["config_version"]
    result["submitted"] = True
    return result
