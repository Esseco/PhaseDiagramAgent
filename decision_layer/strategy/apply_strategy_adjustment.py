"""Apply an already validated adjustable policy patch to runtime policy only."""

from copy import deepcopy

from config_layer.schema.validate_strategy_adjustment import validate_strategy_adjustment


def apply_strategy_adjustment(state: dict, patch: dict, confirmed_config: dict, *, bounds=None, source="agent") -> dict:
    check = validate_strategy_adjustment(patch, confirmed_config, bounds=bounds)
    if not check["valid"]:
        return {"status": "requires_confirmation" if check["requires_user_confirmation"] else "rejected", "state": state, "validation": check}
    updated = deepcopy(state); overlay = updated.setdefault("strategy_overlay", {})
    for path, value in patch.items():
        overlay[path] = deepcopy(value)
    updated.setdefault("strategy_adjustments", []).append({"patch": deepcopy(patch), "source": source, "config_version": updated.get("confirmed_config_version")})
    return {"status": "applied", "state": updated, "validation": check}
