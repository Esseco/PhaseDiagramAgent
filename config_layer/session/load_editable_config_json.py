"""Load a user-edited config JSON and check its shape before draft import."""

from __future__ import annotations

import json
from pathlib import Path

from config_layer.session.create_editable_config_json import FORMAT_ID


_SECRET_KEYS = {"api_key", "token", "password", "secret", "private_key", "access_key"}
_OPTIONAL_FIELDS = {
    "config.system": {"boundary", "phase_reference_directory", "H_generation"},
    "config": {"storage"},
}
_DYNAMIC_OBJECT_PATHS = {
    "config.system.phase_references", "config.system.boundary.H", "config.system.boundary.P",
    "config.generation_actions.quotas", "config.dft.parameters",
    "config.storage.paths",
}
_ATOMIC_OBJECT_PATHS = {"config.system.boundary", *_DYNAMIC_OBJECT_PATHS}


def load_editable_config_json(path, current_config: dict, *, default_storage=None) -> dict:
    """Return the config payload after format, secret, and schema-shape checks."""
    source = Path(path)
    try:
        document = json.loads(
            _strip_jsonc_comments(source.read_text(encoding="utf-8")),
            parse_constant=_reject_nonstandard_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 格式错误：第 {error.lineno} 行第 {error.colno} 列。") from error
    except OSError as error:
        raise ValueError(f"无法读取配置文件：{error}") from error
    from config_layer.session.project_config_json import FORMAT_ID as PROJECT_FORMAT_ID, expand_project_config
    if isinstance(document, dict) and document.get("_format") == PROJECT_FORMAT_ID:
        return expand_project_config(
            document, source=source, baseline_config=current_config,
        )
    if not isinstance(document, dict) or document.get("_format") != FORMAT_ID:
        raise ValueError(f"文件格式标识无效；应为 {FORMAT_ID}。")
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("文件必须包含 config JSON 对象。")
    _check_secrets(config)
    _check_shape(current_config, config, "config")
    config = dict(config)
    if "storage" not in config and default_storage is not None:
        config["storage"] = default_storage
    if "storage" in config:
        from config_layer.session.resolve_workspace_paths import resolve_workspace_paths
        resolve_workspace_paths(config, base_directory=source.parent)
    return config


def config_leaf_patch(old: dict, new: dict, prefix="config") -> dict:
    """Build a minimal dotted-path patch for change-audited draft import."""
    if prefix in _ATOMIC_OBJECT_PATHS and old != new:
        return {prefix.removeprefix("config."): new}
    patch = {}
    for key in sorted(set(old) | set(new)):
        path = f"{prefix}.{key}"
        if key not in old:
            patch[path.removeprefix("config.")] = new[key]
            continue
        before, after = old[key], new[key]
        if isinstance(before, dict) and isinstance(after, dict):
            patch.update(config_leaf_patch(before, after, path))
        elif before != after:
            patch[path.removeprefix("config.")] = after
    return patch


def _check_shape(reference, candidate, path):
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            raise ValueError(f"{path} 必须是 JSON 对象。")
        for key in candidate:
            if not isinstance(key, str) or "." in key or key.startswith("_"):
                raise ValueError(f"{path} 中的字段名无效：{key!r}。")
        missing = set(reference) - set(candidate)
        extra = set(candidate) - set(reference) - _OPTIONAL_FIELDS.get(path, set())
        if path in _DYNAMIC_OBJECT_PATHS:
            missing.clear()
            extra.clear()
        if missing or extra:
            details = []
            if missing:
                details.append("缺少字段：" + ", ".join(sorted(missing)))
            if extra:
                details.append("未知字段：" + ", ".join(sorted(extra)))
            raise ValueError(f"{path} 字段结构不匹配（" + "；".join(details) + "）。不要删除字段；未知值填 null。")
        for key, value in reference.items():
            if key not in candidate:
                continue
            if "." in key:
                raise ValueError(f"配置字段名不能包含点号：{path}.{key}")
            _check_shape(value, candidate[key], f"{path}.{key}")
        return
    if reference is None or candidate is None:
        return
    if isinstance(reference, bool):
        valid = isinstance(candidate, bool)
    elif isinstance(reference, (int, float)):
        valid = isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
    elif isinstance(reference, str):
        valid = isinstance(candidate, str)
    elif isinstance(reference, list):
        valid = isinstance(candidate, list)
    else:
        valid = type(reference) is type(candidate)
    if not valid:
        raise ValueError(f"{path} 类型应与模板一致（当前模板类型：{type(reference).__name__}）。")


def _reject_nonstandard_constant(value):
    raise ValueError(f"不允许非标准 JSON 数值：{value}")


def _strip_jsonc_comments(text):
    """Remove JSONC comments while preserving quoted strings and line numbers."""
    output = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            index += 2
            while index + 1 < len(text) and text[index:index + 2] != "*/":
                if text[index] in "\r\n":
                    output.append(text[index])
                index += 1
            if index + 1 >= len(text):
                raise ValueError("JSONC 块注释未闭合。")
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _check_secrets(value, path="config"):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith(("_password", "_secret", "_token")):
                raise ValueError(f"密钥类字段不允许写入配置 JSON：{path}.{key}")
            _check_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_secrets(child, f"{path}[{index}]")
