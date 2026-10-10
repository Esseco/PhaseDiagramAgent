"""Atomically save draft dialogue or confirmed configuration session."""

import json
from pathlib import Path


def save_config_session(session: dict, path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(session, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return str(output)
