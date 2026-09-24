"""Persist an explicitly confirmed immutable config snapshot in its workspace."""

from __future__ import annotations

import json
from pathlib import Path

from config_layer.session.resolve_workspace_paths import resolve_workspace_paths


def save_confirmed_config_file(snapshot: dict, *, base_directory) -> Path:
    """Create the versioned snapshot once; allow only identical idempotent repeats."""
    version = snapshot.get("config_version")
    if not isinstance(version, str) or not version or any(char in version for char in "/\\"):
        raise ValueError("confirmed snapshot has an invalid config_version")
    config = snapshot.get("config")
    if not isinstance(config, dict):
        raise ValueError("confirmed snapshot has no config object")
    paths = resolve_workspace_paths(config, base_directory=base_directory)
    target = paths["config_snapshots"] / f"{version}.json"
    payload = {
        "config_version": version,
        "config_hash": snapshot.get("config_hash"),
        "config": config,
        "audit": snapshot.get("audit"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except FileExistsError:
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FileExistsError(f"配置快照已存在且无法验证：{target}") from error
        if existing != payload:
            raise FileExistsError(f"配置快照路径已存在不同内容，不会覆盖：{target}")
    return target
