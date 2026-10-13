"""Bounded project-owned startup diagnostics; never read configuration secrets."""

import json
from pathlib import Path


def startup_timeout(settings):
    value = settings.get("agent_startup_timeout_seconds", 90)
    if type(value) not in {int, float} or not 5 <= value <= 600:
        raise ValueError("agent_startup_timeout_seconds 必须在 5–600 秒之间")
    return float(value)


def failure_summary(config_path, fallback_log):
    """Prefer this service attempt's stored error; logs stay locally readable."""
    path = Path(config_path).resolve().parent / "workflow_state" / "studio" / "service.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = {}
    if value.get("launch_log") and value["launch_log"] != str(fallback_log):
        value = {}
    detail = value.get("error") or "服务未能完成初始化"
    child_log = value.get("child_log") or str(fallback_log)
    # Include only exception summaries, never requests or raw config contents.
    try:
        with Path(child_log).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16000))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        root = next(
            (line for line in lines if line.startswith('BlockingError("Blocking call to ')), None
        )
        if root:
            detail += "\n" + root.split("\\n", 1)[0][:600]
        for line in reversed(lines) if not root else []:
            if line.startswith(
                (
                    "ValueError:",
                    "RuntimeError:",
                    "FileNotFoundError:",
                    "PermissionError:",
                    "blockbuster.blockbuster.BlockingError:",
                )
            ):
                detail += "\n" + line[:600]
                break
    except OSError:
        pass
    return f"{detail}\n本次启动日志：{fallback_log}\nStudio 日志：{child_log}"
