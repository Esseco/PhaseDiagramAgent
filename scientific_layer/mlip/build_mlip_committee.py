"""加载具有不同模型文件内容的 MLIP 委员会。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable


def build_mlip_committee(
    model_configs: list[dict[str, Any]],
    *,
    loader: Callable[..., Any] | None = None,
    training_entrypoint: Callable[..., Any] | None = None,
    committee_id: str | None = None,
    data_version: str | None = None,
) -> dict[str, Any]:
    """按文件 SHA256 判重并加载模型；相同内容只保留一个成员。"""
    members, rejected = [], []
    seen = {}
    for index, config in enumerate(model_configs):
        path = Path(config.get("model_path", ""))
        if not path.is_file():
            rejected.append(
                {
                    "index": index,
                    "reason": "model_file_missing",
                    "model_path": str(path),
                }
            )
            continue
        fingerprint = _sha256(path)
        if fingerprint in seen:
            rejected.append(
                {
                    "index": index,
                    "reason": "duplicate_model_content",
                    "duplicate_of": seen[fingerprint],
                    "model_path": str(path),
                }
            )
            continue
        member_id = config.get("model_id", f"MLIP-{fingerprint[:12]}")
        seen[fingerprint] = member_id
        model = None
        status = "not_configured" if loader is None else "loaded"
        error = None
        if loader is not None:
            try:
                model = loader(model_path=path, config=dict(config))
            except Exception as caught:
                status, error = "failed", f"{type(caught).__name__}: {caught}"
        members.append(
            {
                "model_id": member_id,
                "model_path": str(path.resolve()),
                "fingerprint": fingerprint,
                "version": config.get("version"),
                "source": config.get("source"),
                "training_data_version": config.get("training_data_version"),
                "training_run_id": config.get("training_run_id"),
                "status": status,
                "model": model,
                "error": error,
            }
        )
    loaded = [item for item in members if item["status"] == "loaded"]
    return {
        "committee_id": committee_id,
        "data_version": data_version,
        "status": "ready"
        if len(loaded) >= 2
        else "not_configured"
        if loader is None
        else "insufficient_distinct_models",
        "members": members,
        "loaded_members": loaded,
        "rejected": rejected,
        "training_entrypoint": training_entrypoint,
        "training_configured": training_entrypoint is not None,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
