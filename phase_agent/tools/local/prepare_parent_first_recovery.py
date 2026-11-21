"""Register a user-approved parent-first recovery handoff with the Agent."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def prepare_parent_first_recovery(*, action, context):
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.tools.workflows.parent_first_submission import prepare_parent_first_submission

    state = context.get("event_state") or {}
    config = context.get("effective_config") or {}
    params = action.get("parameters") or {}
    manager = context.get("manager")
    approval = context.get("approval_record_id")
    config_version = context.get("config_version")
    root = config.get("upload_batches_directory")
    active_model = state.get("active_model_version")
    if (
        action.get("operation") != "prepare_parent_first_recovery"
        or params.get("mode") != "parent_first_recovery_inputs"
        or params.get("confirmed") is not True
        or not approval
        or not root
        or manager is None
        or not config_version
        or config_version != state.get("confirmed_config_version")
        or not active_model
        or params.get("dft_parameters") != (config.get("dft") or {}).get("parameters", {})
    ):
        return {
            "status": "not_configured",
            "state": state,
            "submitted": False,
            "reason": "parent_first_approval_or_config_invalid",
        }
    epoch = model_epoch(state, active_model)
    if params.get("scope_id") == "initial_direct" and epoch is None:
        epoch = "epoch0"
    if not isinstance(epoch, str) or not epoch.startswith("epoch") or not epoch[5:].isdigit():
        return {
            "status": "not_configured",
            "state": state,
            "submitted": False,
            "reason": "parent_first_epoch_unavailable",
        }
    if params.get("scope_id") == "initial_direct":
        if state.get("dft_dataset_records") or state.get("initial_direct_sampling"):
            return {
                "status": "not_configured",
                "state": state,
                "submitted": False,
                "reason": "initial_direct_already_completed_use_kmeans_recovery",
            }
    elif params.get("extrapolation_policy_version") == "epoch_refresh_v1":
        from phase_agent.agents.reliability.extrapolation_round import required_direct_round
        from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy

        required = required_direct_round(state)
        if not required or epoch_recovery_policy(state, required) != {
            "status": "sampling_required",
            "route": "perturbation",
        }:
            return {
                "status": "not_configured",
                "state": state,
                "submitted": False,
                "reason": "parent_first_epoch_policy_rejected",
            }
    identity = hashlib.sha256(
        json.dumps(
            {"approval": approval, "parameters": params}, sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    previous = (state.get("parent_first_recovery_batches") or {}).get(identity)
    if previous:
        return {
            "status": "not_configured",
            "state": state,
            "submitted": False,
            "reason": "parent_first_existing_batch_requires_reconciliation",
        }
    prefix = "parent-first-"
    if params.get("scope_id") == "initial_direct":
        if state.get("dft_dataset_records") or state.get("initial_direct_sampling"):
            return {
                "status": "not_configured",
                "state": state,
                "submitted": False,
                "reason": "initial_direct_already_completed_use_kmeans_recovery",
            }
    elif params.get("extrapolation_policy_version") == "epoch_refresh_v1":
        from phase_agent.evidence.extrapolation_recovery import recovery_label_facts

        cycle = state.get("extrapolation_recovery_cycle") or {}
        progress = (
            recovery_label_facts(state) if cycle.get("policy_version") == "epoch_refresh_v1" else {}
        )
        from phase_agent.evidence.extrapolation_policy import recovery_lineage

        metadata = recovery_lineage(cycle, active_model, params.get("round_evidence_digest"), epoch)
        prefix = (
            f"extrapolation-round-{metadata['extrapolation_round']:02d}-"
            f"perturbation-attempt-{progress.get('attempt_count', 0) + 1:02d}-"
        )
    target = Path(root) / epoch / "calculations" / (prefix + identity[:16])
    settings = {
        "config_version": config_version,
        "model_version": active_model,
        "approval_record_id": approval,
        "parameters": deepcopy(params["dft_parameters"]),
    }
    mlip_cfg = config.get("mlip") or {}
    direct_cfg = config.get("direct") or {}
    effective_params = deepcopy(params)
    if not effective_params.get("model_sha256") and mlip_cfg.get("model_sha256"):
        effective_params["model_sha256"] = mlip_cfg["model_sha256"]
    f_path = (
        effective_params.get("feature_model_path")
        or direct_cfg.get("feature_model_path")
        or mlip_cfg.get("feature_model_path")
    )
    f_sha = (
        effective_params.get("feature_model_sha256")
        or direct_cfg.get("feature_model_sha256")
        or mlip_cfg.get("feature_model_sha256")
    )
    result = prepare_parent_first_submission(
        state,
        manager.data,
        effective_params,
        target,
        feature_model_path=f_path,
        feature_model_sha256=f_sha,
        dft_settings=settings,
    )
    if result.get("status") != "prepared" or result.get("submitted") is not False:
        raise ValueError("parent_first_submission_preparation_failed")
    current = deepcopy(state)
    batch = {
        "batch_id": identity,
        "status": "prepared",
        "approval_record_id": approval,
        "config_version": config_version,
        "model_version": active_model,
        "scope_id": params["scope_id"],
        "round_evidence_digest": params["round_evidence_digest"],
        "parent_count": result["parent_count"],
        "reference_count": result["reference_count"],
        "target": result["target"],
        "task_id": result["task_id"],
        "input_dir": result["input_dir"],
        "handoff": result["handoff"],
    }
    if params.get("scope_id") == "initial_direct":
        if state.get("dft_dataset_records") or state.get("initial_direct_sampling"):
            return {
                "status": "not_configured",
                "state": state,
                "submitted": False,
                "reason": "initial_direct_already_completed_use_kmeans_recovery",
            }
    elif params.get("extrapolation_policy_version") == "epoch_refresh_v1":
        batch["policy_version"] = "epoch_refresh_v1"
    current.setdefault("parent_first_recovery_batches", {})[identity] = batch
    return {
        "status": "prepared",
        "state": current,
        "submitted": False,
        "parent_first_recovery_batch": batch,
        "handoff": result["handoff"],
    }
