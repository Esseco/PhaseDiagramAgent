"""Filesystem layout for versioned phase-diagram snapshots."""

import hashlib
import re
from pathlib import Path


def phase_snapshot_directory(root, method, model_version=None):
    if method == "mlip":
        version = str(model_version or "unknown")
        safe_version = re.sub(r"[^A-Za-z0-9._-]", "_", version)
        if safe_version != version:
            safe_version += "-" + hashlib.sha256(version.encode()).hexdigest()[:8]
        return Path(root) / safe_version / "history"
    if method == "dft":
        return Path(root) / "dft" / "history"
    raise ValueError("相图方法只能是 mlip 或 dft")
