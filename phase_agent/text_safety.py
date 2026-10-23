"""Bounded credential redaction shared by persistence and presentation."""

import re


def redact_text(value):
    value = re.sub(r"(?i)\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]+", "[已隐藏凭据]", value)

    value = re.sub(
        r"""(?i)((?:api[_ -]?key|password|secret|token|密码)["']?\s*[:=：]\s*["']?)[^\s"']+""",
        r"\1[已隐藏凭据]",
        value,
    )

    return value


def clean(value, depth=0):

    if depth > 8:
        return "…"

    if isinstance(value, dict):
        return {
            str(k): clean(v, depth + 1)
            for k, v in list(value.items())[:40]
            if not any(
                word in str(k).lower()
                for word in (
                    "key",
                    "token",
                    "password",
                    "credential",
                    "secret",
                    "authorization",
                )
            )
        }

    if isinstance(value, (list, tuple)):
        return [clean(v, depth + 1) for v in value[:30]]

    if isinstance(value, str):
        return redact_text(value)[:4000]

    return value if value is None or isinstance(value, (bool, int, float)) else type(value).__name__
