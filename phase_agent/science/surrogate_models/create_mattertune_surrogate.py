"""Reserved MatterTune adapter for structure encoder plus prediction head."""


def create_mattertune_surrogate(config: dict) -> dict:
    if not config.get("enabled") or not callable(config.get("adapter")):
        return {
            "status": "not_implemented",
            "model": None,
            "reason": "MatterTune adapter intentionally left unconfigured",
        }
    return config["adapter"](config=config)
