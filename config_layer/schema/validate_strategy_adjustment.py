"""Allow bounded policy changes; route hard changes back to confirmation."""

from config_layer.session.classify_config_permission import classify_config_permission


def validate_strategy_adjustment(patch: dict, confirmed_config: dict, *, bounds=None) -> dict:
    hard, adjustable, rejected = [], [], []
    for path, value in patch.items():
        permission = classify_config_permission(path, confirmed_config)
        if permission == "hard_constraint":
            hard.append(path); continue
        if permission != "adjustable_policy":
            rejected.append(path); continue
        rule = (bounds or {}).get(path)
        if rule and isinstance(value, (int, float)) and not float(rule[0]) <= value <= float(rule[1]):
            rejected.append(path)
        else:
            adjustable.append(path)
    return {"valid": not hard and not rejected, "auto_execute": not hard and not rejected, "requires_user_confirmation": bool(hard), "hard_changes": hard, "adjustable_changes": adjustable, "rejected": rejected}
