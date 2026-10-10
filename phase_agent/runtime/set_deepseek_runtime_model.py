"""Persist a supported DeepSeek model in the local Open WebUI runtime JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def set_deepseek_runtime_model(runtime_config_path, model: str) -> str:
    """Atomically update only deepseek.model; API secrets are never written."""
    allowed = {"deepseek-flash", "deepseek-v4-pro"}
    if model not in allowed:
        raise ValueError(f"不支持的 DeepSeek 模型：{model}")
    target = Path(runtime_config_path).resolve()
    config = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("运行时配置必须是 JSON object")
    deepseek = config.get("deepseek")
    if not isinstance(deepseek, dict):
        deepseek = {}
        config["deepseek"] = deepseek
    deepseek["model"] = model

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, target)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return model
