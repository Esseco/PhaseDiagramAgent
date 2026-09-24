"""Persist definitions, selection and model configuration (model object optional)."""

import json
from pathlib import Path


def save_surrogate_artifacts(directory, *, feature_definitions: dict, selection: dict | None = None, model_record: dict | None = None) -> dict:
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    payloads = {"feature_definitions.json": feature_definitions, "feature_selection.json": selection or {}, "model_config.json": {key: value for key, value in (model_record or {}).items() if key != "model"}}
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if model_record and model_record.get("model") is not None:
        import joblib
        joblib.dump(model_record["model"], root / "model.joblib")
    return {name: str(root / name) for name in payloads}
