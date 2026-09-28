"""Stable local Open WebUI connection tokens, separate from the DeepSeek key."""

import os
import secrets

from run.deepseek_credentials import _read_windows_credential, _write_windows_credential


def local_service_tokens() -> tuple[str, str]:
    """Use explicit environment tokens or create per-user Windows credentials once."""
    values = []
    for env_name, target in (
        ("OPENWEBUI_TOOL_TOKEN", "PhaseSearchAgent/OpenWebUIToolToken"),
        ("OPENWEBUI_CONTROL_TOKEN", "PhaseSearchAgent/OpenWebUIControlToken"),
    ):
        value = os.environ.get(env_name)
        if not value and os.name == "nt":
            value = _read_windows_credential(target)
            if not value:
                value = secrets.token_urlsafe(32)
                _write_windows_credential(target, value, "Local Phase Search Agent bearer token")
        if not value or len(value) < 16:
            raise RuntimeError(f"{env_name} 未配置；非 Windows 系统请先设置环境变量")
        values.append(value)
    if values[0] == values[1]:
        raise RuntimeError("Open WebUI 连接 token 与控制 token 必须不同")
    return values[0], values[1]
