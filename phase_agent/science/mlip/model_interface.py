"""Explicit local MLIP selection and loading boundary.

The application stores the selected framework/model and an optional local
repository path.  Heavy model imports remain inside the worker environment;
the local repository is a dependency, never a project file to be edited.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys


@contextmanager
def local_repo(path):
    value = str(path or "").strip()
    if not value:
        yield
        return
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"mlip.local_repo_path_not_found:{root}")
    text = str(root)
    inserted = text not in sys.path
    if inserted:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if inserted and text in sys.path:
            sys.path.remove(text)


def model_spec(model: dict | None, *, model_path=None):
    model = dict(model or {})
    framework = str(model.get("framework") or model.get("type") or "mace").lower()
    name = model.get("model_name") or model.get("name") or model.get("version")
    path = model_path or model.get("model_path")
    return {
        "framework": framework,
        "model_name": name,
        "model_path": path,
        "local_repo_path": model.get("local_repo_path"),
        "device": model.get("device", "cpu"),
        "head": model.get("mace_head") or model.get("head"),
        "default_dtype": model.get("mace_default_dtype", "float64"),
    }


def create_calculator(model: dict | None, *, model_path=None):
    spec = model_spec(model, model_path=model_path)
    preset = spec["model_name"] or spec["framework"]
    if not preset:
        raise ValueError("mlip.model_name_required")
    with local_repo(spec["local_repo_path"]):
        try:
            from Process_MLIP.calculators import create_mlip_calculator
        except ModuleNotFoundError as error:
            if (
                error.name != "Process_MLIP"
                or spec["framework"] != "mace"
                or spec["local_repo_path"]
            ):
                raise
            from mace.calculators import MACECalculator

            options = {
                "model_paths": str(spec["model_path"] or ""),
                "device": spec["device"],
                "default_dtype": spec["default_dtype"],
            }
            if spec["head"] is not None:
                options["head"] = spec["head"]
            return MACECalculator(**options)

        # Process_MLIP accepts either a preset or a fully resolved config dict.
        # Use the preset only when no per-run path/framework override exists.
        if spec["framework"] == "mace":
            if spec["model_path"] is None and spec["model_name"]:
                from Process_MLIP.model_config import model_config

                selected = model_config(preset)
                selected["device"] = spec["device"]
            else:
                selected = {
                    "type": "mace",
                    "model_path": str(spec["model_path"] or ""),
                    "device": spec["device"],
                    "head": spec["head"],
                    "default_dtype": spec["default_dtype"],
                    "enable_cueq": False,
                    "enable_symmetrix": False,
                    "symmetrix_use_kokkos": False,
                    "symmetrix_species": None,
                }
        else:
            from Process_MLIP.model_config import model_config

            overrides = (
                {"model_path": str(spec["model_path"])} if spec["model_path"] is not None else {}
            )
            selected = model_config(preset, **overrides)
            if selected["type"] != spec["framework"]:
                raise ValueError("mlip.framework_model_mismatch")
            if "device" in selected:
                selected["device"] = spec["device"]
        return create_mlip_calculator(selected)


def validate_model_selection(config: dict) -> list[str]:
    model = config.get("mlip") or {}
    spec = model_spec(model)
    errors = []
    if spec["framework"] not in {"mace", "sevennet", "chgnet", "orb", "nep"}:
        errors.append(f"mlip.framework_unsupported:{spec['framework']}")
    if spec["framework"] != "mace" and (config.get("mlip_finetune") or {}).get("enabled"):
        errors.append("mlip.finetune_requires_mace")
    if spec["local_repo_path"] and not Path(spec["local_repo_path"]).is_dir():
        errors.append("mlip.local_repo_path_not_found")
    return errors
