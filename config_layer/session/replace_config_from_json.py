"""Replace an unconfirmed in-memory draft with the validated editable file."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


def config_digest(config: dict) -> str:
    data = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def mother_structure_digest(config: dict) -> str:
    references = ((config.get("system") or {}).get("phase_references") or {})
    digest = hashlib.sha256()
    for phase, reference in sorted(references.items()):
        path = reference.get("path") if isinstance(reference, dict) else reference
        source = Path(str(path))
        digest.update(f"{phase}:{source}\0".encode("utf-8"))
        try:
            digest.update(hashlib.sha256(source.read_bytes()).digest())
        except OSError:
            digest.update(b"<unavailable>")
    return digest.hexdigest()


def replace_config_from_json(session: dict, config: dict, *, source_path: str,
                             source_hash: str) -> dict:
    if session.get("status") != "draft":
        raise ValueError("已确认的配置不能由草稿文件替换")
    updated = deepcopy(session)
    changed = updated.get("config") != config
    if changed:
        updated["config"] = deepcopy(config)
        updated["draft_revision"] = int(updated.get("draft_revision", 0)) + 1
    updated["last_imported_config_hash"] = source_hash
    updated["last_imported_config_path"] = source_path
    updated["last_imported_config_digest"] = config_digest(config)
    updated["last_imported_mother_digest"] = mother_structure_digest(config)
    updated.pop("agent_reviewed_revision", None)
    updated.pop("agent_reviewed_config_hash", None)
    updated.pop("agent_reviewed_config_digest", None)
    updated.pop("agent_reviewed_mother_digest", None)
    # The imported file is now authoritative.  Any patch proposed against an
    # older file hash must not survive the import and be applied later by a
    # bare "写入" command.
    updated.pop("pending_config_patch", None)
    updated.setdefault("dialogue", []).append({
        "type": "config_revision" if changed else "config_import",
        "author": "user_json_draft",
        "revision": updated.get("draft_revision"),
        "source_path": source_path,
        "source_hash": source_hash,
        "reason": "以已读取、展开和校验的配置文件替换未确认草稿；清除旧会话额外字段。",
    })
    return updated
