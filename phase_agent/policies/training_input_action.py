"""Identify input-only proposals without weakening model-activation approval."""


def is_training_input_action(action):
    parameters = (action or {}).get("parameters") or {}
    tool = (action or {}).get("tool")
    return (
        tool == "update_mlip"
        and parameters.get("prepare_inputs_only") is True
        and parameters.get("action", "RETRAIN_MLIP") == "RETRAIN_MLIP"
    ) or (
        tool == "adjust_strategy"
        and parameters.get("request_configuration_revision") is True
        and parameters.get("patch") == {"mlip_finetune.enabled": True}
    )
