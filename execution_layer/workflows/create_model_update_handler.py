"""Compose the existing train, validate, activate, and reevaluate interfaces."""

from copy import deepcopy

from data_layer.models.activate_validated_model import activate_validated_model
from scientific_layer.training.update_mlip import update_mlip
from scientific_layer.training.validate_mlip import validate_mlip


def create_model_update_handler(
    *,
    trainer=None,
    validation_evaluator=None,
    reevaluation_predictor=None,
    historical_data_provider=None,
    validation_data_provider=None,
    candidate_provider=None,
):
    """Create the callback accepted by ``run_active_learning_cycle``.

    External ML frameworks remain adapters. A candidate model is activated only
    after independent validation passes.
    """
    def handler(*, trigger, state, manager, config):
        current = deepcopy(state)
        if trigger.get("action") == "REJECT_CANDIDATE_MODEL":
            version = trigger.get("candidate_model_version")
            stored = (current.get("candidate_models") or {}).get(version)
            reason = str(trigger.get("user_rejection_reason") or "").strip()
            if not stored:
                return {"status": "rejection_rejected", "reason": "candidate_model_unknown", "state": current}
            if not reason:
                return {"status": "awaiting_rejection_reason", "state": current}
            stored["status"] = "user_rejected"
            stored["user_rejection_reason"] = reason
            current.setdefault("model_candidate_review_history", []).append({
                "candidate_model_version": version, "decision": "rejected", "reason": reason,
                "old_model_version": stored.get("old_model_version"),
                "dft_data_version": stored.get("dft_data_version"),
                "validation_data_version": stored.get("validation_data_version")})
            return {"status": "candidate_rejected_by_user", "state": current,
                    "validation": deepcopy(stored.get("validation"))}
        if trigger.get("action") == "ACTIVATE_CANDIDATE_MODEL":
            version = trigger.get("candidate_model_version")
            stored = (current.get("candidate_models") or {}).get(version)
            if not stored:
                return {"status": "activation_rejected", "reason": "candidate_model_unknown", "state": current}
            activation = activate_validated_model(
                current, stored["model"], stored["validation"],
                approval_reason=trigger.get("user_approval_reason"),
            )
            if activation["status"] == "activated":
                activated_state = activation["state"]
                activated_state["active_model"] = deepcopy(stored["model"])
                activated_state["last_trained_dataset_version"] = stored.get("dft_data_version")
                activated_state["new_dft_records"] = []
                _append_model_epoch(activated_state, stored["model"], stored["validation"],
                                    stored.get("old_model_version"))
                activation["reevaluation"] = {"status": "awaiting_refresh_approval"}
            return {"status": activation["status"], "activation": activation,
                    "state": activation["state"], "validation": stored["validation"]}
        iteration = int(current.get("iteration", 0))
        dataset_version = f"dft-data-{iteration:06d}"
        candidate_version = f"mlip-candidate-{iteration:06d}"
        old_model = deepcopy(current.get("active_model") or config.get("mlip") or {})
        from scientific_layer.training.cumulative_training_records import cumulative_training_records
        historical = (_provide(historical_data_provider, state=current, manager=manager, config=config) or []
                      if callable(historical_data_provider) else cumulative_training_records(current))
        finetune_config = deepcopy(config.get("mlip_finetune") or {})
        training_parameters = finetune_config.setdefault("training", {})
        if not training_parameters.get("foundation_model"):
            training_parameters["foundation_model"] = old_model.get("model_path")
        finetune_config["main_model_index"] = 0
        training = update_mlip(
            list(current.get("new_dft_records") or []),
            historical_data=historical,
            trainer=trainer,
            base_model=old_model,
            training_config=finetune_config,
            dataset_version=dataset_version,
            candidate_model_version=candidate_version,
        )
        result = {"status": training["status"], "trigger": deepcopy(trigger), "training": training, "validation": None, "activation": None, "reevaluation": None, "state": current}
        if training["status"] != "trained_candidate":
            return result
        candidate_model = deepcopy(training["candidate_model"] or {})
        candidate_model.setdefault("version", candidate_version)
        candidate_model["dataset_version"] = dataset_version
        candidate_model["main_model_index"] = 0
        if not candidate_model.get("model_paths"):
            members = candidate_model.get("members") or candidate_model.get("committee") or []
            paths = [item.get("model_path") for item in members
                     if isinstance(item, dict) and item.get("model_path")]
            if paths:
                candidate_model["model_paths"] = paths
        validation_data = _provide(validation_data_provider, state=current, manager=manager, config=config) or []
        validation = validate_mlip(
            old_model,
            candidate_model,
            validation_data,
            evaluator=validation_evaluator,
            criteria=deepcopy((config.get("mlip_finetune") or {}).get("validation") or {}),
            validation_data_version=f"validation-{dataset_version}",
        )
        result["validation"] = validation
        current.setdefault("candidate_models", {})[candidate_model["version"]] = {
            "model": deepcopy(candidate_model), "old_model": deepcopy(old_model),
            "old_model_version": old_model.get("version"),
            "validation": deepcopy(validation), "training_summary": deepcopy(training.get("metadata")),
            "dft_data_version": dataset_version,
            "validation_data_version": validation.get("validation_data_version"),
            "anomalies": deepcopy(validation.get("anomalies") or validation.get("missing_metrics") or []),
            "status": "validated_candidate" if validation.get("passed") else "validation_blocked_candidate",
        }
        result["state"] = current
        if not validation.get("passed"):
            result["status"] = "validation_rejected" if validation.get("status") == "completed" else validation.get("status", "failed")
            return result
        if validation.get("benefit_status") == "no_distinguishable_benefit":
            result["status"] = "candidate_retained_no_clear_benefit"
            return result
        activation_settings = (config.get("mlip_finetune") or {}).get("activation") or {}
        if activation_settings.get("requires_separate_approval"):
            result["status"] = "awaiting_activation_approval"
            return result
        activation = activate_validated_model(current, candidate_model, validation,
                                              approval_reason=trigger.get("user_approval_reason"))
        result["activation"] = activation
        if activation.get("status") != "activated":
            result["status"] = "activation_rejected"
            return result
        current = activation["state"]
        current["active_model"] = candidate_model
        current["last_trained_dataset_version"] = dataset_version
        _append_model_epoch(current, candidate_model, validation, old_model.get("version"))
        current["new_dft_records"] = []
        reevaluation = {"status": "awaiting_refresh_approval"}
        result.update({"status": "activated", "reevaluation": reevaluation, "state": current})
        return result

    return handler


def _provide(provider, **context):
    return provider(**context) if callable(provider) else None


def _append_model_epoch(state, model, validation, previous_version):
    version = model.get("version")
    rows = state.setdefault("model_update_epochs", [])
    if any(row.get("model_version") == version for row in rows):
        return
    rows.append({"epoch": len(rows) + 1, "model_version": version,
                 "previous_model_version": previous_version,
                 "training_data_version": model.get("dataset_version"),
                 "validation_data_version": validation.get("validation_data_version"),
                 "energy_mae": (validation.get("new_metrics") or {}).get("energy_mae"),
                 "hull_change": None, "ground_state_unchanged": None})


