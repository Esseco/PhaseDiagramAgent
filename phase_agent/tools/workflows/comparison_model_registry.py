"""Keep per-version model references/fingerprints for original-round DFT checks."""

from copy import deepcopy
from pathlib import Path

from phase_agent.tools.remote.integrity import file_checksum


def comparison_model_path(model):
    path = model.get("local_model_path") or model.get("model_path")
    if not path and model.get("model_paths"):
        path = model["model_paths"][int(model.get("main_model_index", 0))]
    return path


def remember_comparison_models(state, config):
    """Store the initial model as well as activated historical models, by version."""
    registry = state.setdefault("model_registry", {})
    configured = deepcopy(config.get("mlip") or {})
    active = deepcopy(state.get("active_model") or {})
    if not active and configured:
        state["active_model"] = deepcopy(configured)
    models = [configured, active]
    explicit = configured.get("comparison_models") or {}
    for version, model in explicit.items():
        models.append({**deepcopy(model), "version": version})
    for model in models:
        version = model.get("version") or model.get("name")
        if not version:
            continue
        model["version"] = version
        entry = registry.setdefault(
            version, {"model": deepcopy(model), "status": "comparison_reference"}
        )
        frozen = entry.setdefault("comparison_model", deepcopy(entry.get("model") or model))
        frozen.setdefault("version", version)
        path = comparison_model_path(model)
        if not path or not Path(path).is_file():
            continue
        signature = _signature(path)
        cached = entry.get("comparison_file_signature")
        digest = entry.get("comparison_model_sha256")
        if cached != signature or not digest:
            actual = file_checksum(path)
            if digest and digest != actual:
                # Same version cannot silently become a different model.
                entry["comparison_binding_error"] = "model content changed under the saved version"
                continue
            digest = actual
        frozen["local_model_path"] = str(path)
        entry["comparison_model_sha256"] = digest
        entry["comparison_file_signature"] = signature
        entry.pop("comparison_binding_error", None)
    return state


def original_round_model(state, config, version):
    entry = (state.get("model_registry") or {}).get(version) or {}
    if entry.get("comparison_binding_error"):
        raise ValueError(entry["comparison_binding_error"])
    model = deepcopy(entry.get("comparison_model") or entry.get("model") or {})
    if not model:
        configured = config.get("mlip") or {}
        if (configured.get("version") or configured.get("name")) == version:
            model = deepcopy(configured)
    if not version or not model:
        raise ValueError("DFT task model version unavailable; cannot use the current model instead")
    declared = model.get("version") or model.get("name") or version
    if declared != version:
        raise ValueError("historical comparison model version mismatch")
    return model, entry.get("comparison_model_sha256"), entry.get("comparison_file_signature")


def verified_model_digest(path, expected, signature):
    """One stat check per prediction; hash only when the saved file signature differs."""
    actual = expected if expected and signature == _signature(path) else file_checksum(path)
    if expected and actual != expected:
        raise ValueError("original-round model fingerprint mismatch")
    return actual


def _signature(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
