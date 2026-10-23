"""Runtime configuration reading, reference resolution and restart inspection."""

from copy import deepcopy
from phase_agent.module_references import import_saved_module
import json
from pathlib import Path


def _has_history(state, manager) -> bool:
    ledger = getattr(manager, "data", {}) or {}
    if ledger.get("branches") or ledger.get("structures"):
        return True
    keys = (
        "action_records",
        "decisions",
        "tasks",
        "pending_tasks",
        "slurm_batches",
        "event_history",
        "pending_execution_policies",
    )
    return any(state.get(key) for key in keys)


def _resolve_path(value, base):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def runtime_base_directory(config_file, settings):
    """Resolve the existing default-diagram alias identically for run and diagnosis."""
    base = Path(config_file).resolve().parent
    if (settings.get("shared_dft") or {}).get(
        "diagram_id"
    ) == "diagram_default" and base.parent.name == "diagrams":
        system_root = base.parent.parent
        descriptor = system_root / "system.json"
        if (
            descriptor.is_file()
            and _load_json_object(descriptor, "system").get("default_diagram") == base.name
        ):
            return system_root
    return base


def _load_json_object(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ValueError(f"{label}不存在：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} JSON 无效：{path}，第 {error.lineno} 行第 {error.colno} 列；"
            "修正文件后重新读取"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是 JSON object：{path}")
    return value


def _import_reference(reference, label):
    module_name, separator, name = str(reference or "").partition(":")
    if not separator:
        raise ValueError(f"{label}必须使用 package.module:function")
    value = getattr(import_saved_module(module_name), name)
    return value() if callable(value) else value


def _reject_secrets(config):
    forbidden = {"api_key", "token", "password", "secret", "private_key"}
    found = []

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                field = str(key).lower()
                path = f"{prefix}.{key}" if prefix else str(key)
                if field in forbidden or field.endswith("_password") or field.endswith("_secret"):
                    found.append(path)
                walk(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")

    walk(config)
    if found:
        raise ValueError("运行时 JSON 不得包含密钥字段：" + ", ".join(found))


def _session_from_state(state):
    config = state.get("confirmed_config")
    version = state.get("confirmed_config_version") or state.get("config_version")
    if not isinstance(config, dict) or not version:
        return None
    return {
        "status": "confirmed",
        "config": deepcopy(config),
        "dialogue": [],
        "confirmed_snapshot": {
            "config_version": str(version),
            "config_hash": state.get("confirmed_config_hash"),
            "config": deepcopy(config),
            "audit": {"valid": True, "errors": []},
        },
    }
