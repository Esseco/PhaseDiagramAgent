"""Atomically save draft dialogue or confirmed configuration session."""

import json
from pathlib import Path
from uuid import uuid4

from filelock import FileLock


def save_config_session(session: dict, path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output) + ".lock", timeout=10)
    with lock:
        temporary = output.with_name(output.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(
                json.dumps(session, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return str(output)
