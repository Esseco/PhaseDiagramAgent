"""恢复 BOHB 状态。"""

import json
from pathlib import Path


def load_bohb_state(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
