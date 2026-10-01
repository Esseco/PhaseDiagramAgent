"""Role-specific client parameters; no credentials, state changes or I/O."""

CONFIG_INTENT_PROMPT = (
    "Classify only the current user's direct intent to change a persistent "
    "project parameter. A direct request to set/change a value authorizes "
    "a draft edit even without the words config file. Return JSON with "
    "intent and direct_request. Never infer permission from history."
)


def search_client_settings(settings):
    return {
        "model": settings.get("model", "deepseek-v4-pro"),
        "base_url": settings.get("base_url", "https://api.deepseek.com"),
        "max_tokens": int(settings.get("max_tokens", 800)),
        "timeout": int(settings.get("timeout", 60)),
        "thinking": settings.get("thinking"),
        "routine_max_tokens": int(settings.get("routine_max_tokens", 1600)),
        "reasoning_max_tokens": int(settings.get("reasoning_max_tokens", 8192)),
    }


def configuration_client_settings(settings, *, system_prompt):
    parameters = {"model": settings.get("model", "deepseek-v4-pro"),
                  "base_url": settings.get("base_url", "https://api.deepseek.com"),
                  "max_tokens": int(settings.get("max_tokens", 800)),
                  "timeout": int(settings.get("timeout", 60))}
    parameters.update(system_prompt=system_prompt,
                      thinking=settings.get("configuration_thinking", "disabled"))
    return parameters


def intent_client_settings(settings, *, system_prompt):
    return {
        "model": settings.get("model", "deepseek-flash"),
        "base_url": settings.get("base_url", "https://api.deepseek.com"),
        "max_tokens": 120,
        "timeout": int(settings.get("timeout", 60)),
        "thinking": "disabled",
        "system_prompt": system_prompt,
    }
