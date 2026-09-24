"""Recognize explicit local DeepSeek model-selection requests."""

from __future__ import annotations

import re


def resolve_deepseek_model_request(message: str) -> str | None:
    """Return a supported canonical model ID only for an explicit switch request."""
    text = " ".join(str(message or "").strip().lower().split())
    compact = re.sub(r"[\s_.-]+", "", text)
    exact_selection = compact in {
        "v4flash", "v41flash", "deepseekflash", "deepseekv4flash", "deepseekv41flash",
        "v4pro", "deepseekv4pro",
    }
    switch_intent = exact_selection or any(word in text for word in (
        "切换", "切回", "换成", "改成", "改为", "设置为", "使用", "switch", "set model", "/model",
    ))
    if not switch_intent:
        return None
    if "v4.1 flash" in text or "v4 flash" in text or "deepseek-flash" in text or "deepseekv4flash" in compact:
        return "deepseek-flash"
    if "v4 pro" in text or "deepseek-v4-pro" in text or "deepseekv4pro" in compact:
        return "deepseek-v4-pro"
    return None
