"""Active-learning entry point bound to a confirmed configuration version."""

from run.run_active_learning_cycle import run_active_learning_cycle
from execution_layer.dispatch.create_tool_registry import create_tool_registry


def run_confirmed_active_learning_cycle(
    manager,
    phase_references,
    run_config,
    config_session,
    **kwargs,
):
    snapshot = config_session.get("confirmed_snapshot") or {}
    if config_session.get("status") != "confirmed" or not snapshot:
        return {
            "status": "rejected",
            "reason": "configuration_not_confirmed",
            "submitted": False,
        }
    requested_version = kwargs.pop("config_version", None)
    confirmed_version = snapshot["config_version"]
    if requested_version is not None and requested_version != confirmed_version:
        return {
            "status": "rejected",
            "reason": "config_version_mismatch",
            "submitted": False,
            "config_version": confirmed_version,
        }
    result = run_active_learning_cycle(
        manager,
        phase_references,
        run_config,
        config_version=confirmed_version,
        config_session=config_session,
        tool_registry=kwargs.pop("tool_registry", None) or create_tool_registry(),
        **kwargs,
    )
    result["config_version"] = confirmed_version
    result["submitted"] = result.get("status") not in {"rejected", "rejected_by_user"}
    return result
