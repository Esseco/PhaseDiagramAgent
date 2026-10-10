"""Atomic JSON helpers shared by the step runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def read_json(path, default=None):
    source = Path(path)
    if not source.is_file():
        return default
    return json.loads(source.read_text(encoding="utf-8"))


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    if (
        target.is_file()
        and target.stat().st_size == len(encoded)
        and target.read_bytes() == encoded
    ):
        return str(target)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(3):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
    return str(target)


def content_id(payload, prefix="record"):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"
