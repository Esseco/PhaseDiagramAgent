"""Extract information available before the deep-search decision."""

from copy import deepcopy


FORBIDDEN_KEYS = {
    "deep_search_energy", "minimum_energy", "minimum_energy_per_atom", "best_structure",
    "mc_result", "ehull", "dft_energy", "target", "label",
}


def extract_selection_features(branch: dict, structures: list[dict], config: dict) -> dict:
    fields = config.get("feature_fields") or []
    features = {key: deepcopy(branch.get(key)) for key in fields}
    provenance = {key: "branch_ledger_before_selection" for key in fields}
    missing = [key for key, value in features.items() if value is None]
    if config.get("allow_config_features", True):
        values = []
        for structure in structures:
            item = structure.get("config_features")
            if item is not None:
                values.append(deepcopy(item))
        if values:
            features["initial_config_features"] = values
            provenance["initial_config_features"] = "structure.config_features"
    leaked = sorted(_find_forbidden(features))
    return {
        "status": "invalid" if leaked else "completed",
        "features": features,
        "provenance": provenance,
        "missing": missing,
        "leakage_fields": leaked,
    }


def _find_forbidden(value, prefix="") -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_KEYS:
                found.add(path)
            found.update(_find_forbidden(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_find_forbidden(child, f"{prefix}[{index}]"))
    return found
