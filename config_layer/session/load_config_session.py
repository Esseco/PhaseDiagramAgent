"""Load a saved dialogue/configuration session."""

import json
from pathlib import Path


def load_config_session(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
