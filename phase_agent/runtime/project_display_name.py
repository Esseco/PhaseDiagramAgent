"""Conversation labels independent of scientific settings."""

import json
import re
from pathlib import Path


def project_root(state_path):
    path = Path(state_path).absolute()
    return (path.parent.parent if path.parent.name == "workflow_state" else path.parent).resolve()


def display_name(root):
    root = Path(root)
    metadata = root / "project_display.json"
    if metadata.is_file():
        value = json.loads(metadata.read_text(encoding="utf-8")).get("display_name")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if root.parent.name == "diagrams":
        system = root.parent.parent.name
        return system if root.name == "diagram_default" else f"{system} · {root.name}"
    return root.name


def project_name_reply(message, state_path):
    text = str(message).strip()
    root = project_root(state_path)
    if text in {"项目名称", "项目名称是什么", "项目叫什么", "查看项目名称"}:
        return "项目名称：" + display_name(root)
    match = re.fullmatch(
        r"(?:设置|修改|更改)?\s*项目(?:名称|名)(?:\s*[:：=]\s*|\s*(?:设为|改为|叫|是)\s*)(.+)", text
    )
    if not match:
        match = re.fullmatch(r"(?:给)?项目(?:起名为|命名为|改名为|叫)\s*(.+)", text)
    if not match:
        return None
    name = match.group(1).strip().strip("“”\"'")
    if not name or len(name) > 40 or any(ord(c) < 32 for c in name):
        return "项目名称请使用 1–40 个字符，保持单行。"
    metadata = root / "project_display.json"
    data = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
    data["display_name"] = name
    root.mkdir(parents=True, exist_ok=True)
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata)
    return f"项目名称已设为“{name}”。"
