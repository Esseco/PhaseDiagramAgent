"""Validate explicit Windows-local to Linux-remote path mappings."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
import re


def validate_path_mappings(mappings) -> list[dict]:
    if not isinstance(mappings, list):
        raise ValueError("path_mappings must be a list")
    normalized = []
    for index, row in enumerate(mappings):
        if not isinstance(row, dict):
            raise ValueError(f"path_mappings[{index}] must be an object")
        local = str(row.get("windows_local") or "").strip()
        remote = str(row.get("linux_remote") or "").strip()
        if not _windows_absolute(local):
            raise ValueError(
                f"path_mappings[{index}].windows_local must be an absolute Windows path"
            )
        if not PurePosixPath(remote).is_absolute() or _windows_absolute(remote):
            raise ValueError(f"path_mappings[{index}].linux_remote must be an absolute Linux path")
        normalized.append(
            {
                "windows_local": str(PureWindowsPath(local)),
                "linux_remote": str(PurePosixPath(remote)),
            }
        )
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            left_root, right_root = (
                PureWindowsPath(left["windows_local"]),
                PureWindowsPath(right["windows_local"]),
            )
            if _contains(left_root, right_root) or _contains(right_root, left_root):
                raise ValueError("overlapping Windows path mappings are ambiguous")
    return normalized


def map_windows_to_linux(path, mappings) -> str:
    source = PureWindowsPath(str(path))
    if not source.is_absolute():
        raise ValueError("local path must be an absolute Windows path")
    for row in validate_path_mappings(mappings):
        root = PureWindowsPath(row["windows_local"])
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        return str(PurePosixPath(row["linux_remote"], *relative.parts))
    raise ValueError("no confirmed Windows-Linux path mapping covers this path")


def reject_windows_path_in_remote_text(value: str) -> None:
    if re.search(r"(?i)(?:^|[\s='\"])[a-z]:[\\/]", str(value)):
        raise ValueError("Windows absolute paths are forbidden in remote scripts")


def _windows_absolute(value):
    return bool(re.match(r"(?i)^[a-z]:[\\/]", str(value)))


def _contains(parent, child):
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
