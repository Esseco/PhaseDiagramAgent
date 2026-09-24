"""Record the one-time default-parameter choice and optional selective overrides."""
from copy import deepcopy

from config_layer.session.apply_config_revision import apply_config_revision


def answer_default_parameter_prompt(session: dict, *, use_defaults: bool,
                                    overrides: dict | None = None) -> dict:
    prompt = session.get("default_parameter_prompt") or {}
    if prompt.get("status") == "answered":
        raise ValueError("default parameter question has already been answered")
    updated = deepcopy(session)
    updated["default_parameter_prompt"] = {
        **deepcopy(prompt), "status": "answered", "use_defaults": bool(use_defaults),
    }
    updated.setdefault("dialogue", []).append({
        "type": "default_parameter_answer", "role": "user",
        "use_defaults": bool(use_defaults),
    })
    if not use_defaults:
        updated["requires_full_configuration"] = True
        return updated
    if overrides:
        updated = apply_config_revision(
            updated, overrides,
            reasons={path: "user selective default override" for path in overrides},
            author="user",
        )
    collection = deepcopy(updated["config"])
    updated["default_parameter_collection"] = collection
    updated["dialogue"].append({
        "type": "default_parameter_collection",
        "role": "assistant",
        "message": "已采用默认参数。以下为完整参数合集；可按 JSON 路径选择性修改。",
        "parameters": deepcopy(collection),
    })
    return updated
