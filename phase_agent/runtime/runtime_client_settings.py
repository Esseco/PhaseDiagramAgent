"""Role-specific client parameters; no credentials, state changes or I/O."""


def search_client_settings(settings):
    return {
        "proposal_harness": settings.get("proposal_harness", "legacy"),
        "deepagents_max_tokens": int(settings.get("deepagents_max_tokens", 4096)),
        "deepagents_recursion_limit": int(settings.get("deepagents_recursion_limit", 8)),
        "model": settings.get("model", "deepseek-v4-pro"),
        "base_url": settings.get("base_url", "https://api.deepseek.com"),
        "max_tokens": int(settings.get("max_tokens", 800)),
        "timeout": int(settings.get("timeout", 60)),
        "thinking": settings.get("thinking"),
        "routine_max_tokens": int(settings.get("routine_max_tokens", 1600)),
        "reasoning_max_tokens": int(settings.get("reasoning_max_tokens", 8192)),
    }


def configuration_client_settings(settings, *, system_prompt):
    parameters = {
        "model": settings.get("model", "deepseek-v4-pro"),
        "base_url": settings.get("base_url", "https://api.deepseek.com"),
        "max_tokens": int(settings.get("max_tokens", 800)),
        "timeout": int(settings.get("timeout", 60)),
    }
    parameters.update(
        system_prompt=system_prompt, thinking=settings.get("configuration_thinking", "disabled")
    )
    return parameters
