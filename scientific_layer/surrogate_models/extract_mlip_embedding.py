"""Reserved embedding adapter; missing configuration is explicit."""


def extract_mlip_embedding(structure, *, config: dict, context=None) -> dict:
    if not config.get("enabled") or not callable(config.get("extractor")):
        return {"status": "not_configured", "values": None, "version": config.get("version"), "error": "embedding extractor not configured"}
    try:
        values = config["extractor"](structure=structure, context=context or {})
        return {"status": "completed", "values": list(values), "version": config.get("version"), "error": None}
    except Exception as error:
        return {"status": "failed", "values": None, "version": config.get("version"), "error": f"{type(error).__name__}: {error}"}
