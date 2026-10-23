"""Recover the optional launcher index, never configuration or scientific state."""

from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path

from filelock import FileLock
from phase_agent.tools.step_runner.file_protocol import _write_json


def _decode(raw):
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError):
        return [], True
    if not isinstance(value, list):
        return [], True
    valid = list(
        dict.fromkeys(
            row
            for row in value
            if isinstance(row, str)
            and row.strip()
            and not any(ord(c) < 32 or ord(c) == 127 for c in row)
        )
    )
    return valid, value != valid


def _read_locked(path):
    if not path.is_file():
        return []
    raw = path.read_bytes()
    projects, invalid = _decode(raw)
    if not invalid:
        return projects
    backup = path.with_name(f"{path.stem}.invalid-{hashlib.sha256(raw).hexdigest()[:12]}.json")
    try:
        with backup.open("xb") as stream:
            try:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                backup.unlink(missing_ok=True)
                raise
    except FileExistsError:
        if backup.read_bytes() != raw:
            raise ValueError(f"项目清单备份内容冲突；原清单未改：{backup}") from None
    _write_json(path, projects)
    _write_json(
        path.with_suffix(".recovery.json"),
        {"backup": str(backup), "retained_entries": len(projects)},
    )
    logging.getLogger(__name__).warning(
        "项目入口清单已恢复，保留 %s 项；原文件备份：%s。可重新选择项目文件夹添加；科学数据未修改。",
        len(projects),
        backup,
    )
    return projects


def read_project_registry(path):
    path = Path(path).resolve()
    if not path.is_file():
        return []
    projects, invalid = _decode(path.read_bytes())
    if not invalid:
        return projects
    with FileLock(str(path) + ".lock", timeout=5):
        return _read_locked(path)


@contextmanager
def edit_project_registry(path):
    """Serialize the complete read/validate/update; publish no partial index."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=5):
        before = _read_locked(path)
        projects = list(before)
        yield projects
        if projects != before or not path.is_file():
            _write_json(path, projects)


def registry_recovery_notice(path):
    try:
        value = json.loads(Path(path).with_suffix(".recovery.json").read_text(encoding="utf-8"))
        backup = value["backup"]
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    return f"项目入口清单已恢复，原文件保存在 {backup}。列表中缺少的项目可用‘选择文件夹’重新添加；项目配置和科学数据未修改。"
