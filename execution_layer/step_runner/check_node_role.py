"""Prevent accidental use of networked commands on offline workers."""

import os


def check_node_role(required: str, *, role: str | None = None) -> dict:
    actual = (role or os.getenv("PHASEDIAGRAM_NODE_ROLE") or "unspecified").lower()
    allowed = required == "login_or_compute" or actual == "unspecified" or actual == required
    if not allowed:
        raise RuntimeError(f"command requires node_role={required}, current={actual}")
    return {"required": required, "actual": actual, "validated": actual != "unspecified"}
