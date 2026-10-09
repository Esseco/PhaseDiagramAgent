"""Validate a workspace-relative editable config location, including subfolders."""
from pathlib import Path


def editable_config_location(root, setting):
    relative = Path(setting)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".json":
        raise ValueError("可编辑配置必须是工作区内的相对 JSON 路径")
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("可编辑配置不能越出工作区")
    return target
