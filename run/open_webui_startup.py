"""Keep the local Open WebUI startup settings independent of search projects."""

from __future__ import annotations

import json
import os
from pathlib import Path


STARTUP_KEYS = (
    "open_webui_url",
    "open_webui_start_command",
    "open_webui_workdir",
    "open_webui_startup_timeout_seconds",
    "open_webui_environment",
)


def startup_settings_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("找不到 LOCALAPPDATA，无法保存本机 Open WebUI 启动设置")
    return Path(local_app_data) / "PhaseSearchAgent" / "open_webui_startup.json"


def load_startup_settings(path: Path | None = None) -> dict:
    target = path or startup_settings_path()
    if not target.is_file():
        return {}
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Open WebUI 启动设置必须是 JSON 对象：{target}")
    return {key: data[key] for key in STARTUP_KEYS if key in data}


def save_startup_settings(settings: dict, path: Path | None = None) -> Path:
    """Save only non-secret desktop startup settings, once for all projects."""
    target = path or startup_settings_path()
    command = settings.get("open_webui_start_command")
    if command is not None and (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or not arg for arg in command)
    ):
        raise ValueError("Open WebUI 启动命令必须是非空 JSON 字符串数组")
    environment = settings.get("open_webui_environment") or {}
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        or any(secret in key.upper() for secret in ("SECRET", "TOKEN", "PASSWORD", "API_KEY"))
        for key, value in environment.items()
    ):
        raise ValueError("Open WebUI 共享环境变量必须是普通字符串设置，不能保存密钥")
    data = {key: settings[key] for key in STARTUP_KEYS if key in settings and settings[key] is not None}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def effective_webui_config(project_config: dict, path: Path | None = None) -> dict:
    """An explicit project setting overrides the shared desktop setting."""
    return {**load_startup_settings(path), **project_config}
