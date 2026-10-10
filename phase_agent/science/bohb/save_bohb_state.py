"""原子保存 BOHB 状态，支持暂停后恢复。"""

import json
from pathlib import Path


def save_bohb_state(state: dict, path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(output)
    return output
