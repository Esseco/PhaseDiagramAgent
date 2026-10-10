"""Load a previously saved round scheduler state."""

import json
from pathlib import Path


def load_round_scheduler_state(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
