"""Filesystem layout for versioned phase-diagram snapshots."""

import hashlib
import re
from pathlib import Path


def model_output_directory(root, model_version=None, *, state=None):
    if state is not None:
        from phase_agent.tools.remote.model_upload_directory import model_upload_directory

        return model_upload_directory(root, state, model_version)
    version = str(model_version or "unknown")
    if version in {".", ".."}:
        raise ValueError("invalid model output version")
    safe_version = re.sub(r"[^A-Za-z0-9._-]", "_", version)
    if safe_version != version:
        safe_version += "-" + hashlib.sha256(version.encode()).hexdigest()[:8]
    return Path(root) / safe_version


def phase_snapshot_directory(root, method, model_version=None, *, state=None):
    if state is not None:
        if method not in {"mlip", "dft", "combined"}:
            raise ValueError("相图方法只能是 mlip、dft 或 combined")
        version = model_version or state.get("active_model_version")
        if not version:
            raise ValueError("缺少相图输出所属模型版本，不能推测 epoch")
        return (
            model_output_directory(root, version, state=state)
            / "phase_diagrams"
            / method
            / "history"
        )
    if method == "mlip":
        return model_output_directory(root, model_version) / "phase_diagrams" / "history"
    if method == "dft":
        return Path(root) / "dft" / "phase_diagrams" / "history"
    if method == "combined":
        return (
            model_output_directory(root, model_version) / "combined" / "phase_diagrams" / "history"
        )
    raise ValueError("相图方法只能是 mlip 或 dft")
