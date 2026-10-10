"""Export approved user preferences for reviewable cross-project reuse."""

import json
from pathlib import Path


def publish_user_preferences(state, library_root, *, approved):
    if approved is not True:
        raise ValueError("explicit user approval is required")
    records = [
        row
        for row in ((state.get("decision_memory") or {}).get("records") or [])
        if row.get("scope") == "user"
        and row.get("status") == "active"
        and row.get("approved_by_user") is True
    ]
    root = Path(library_root).resolve() / "user_preferences"
    root.mkdir(parents=True, exist_ok=True)
    version = int((state.get("decision_memory") or {}).get("version") or 0)
    path = root / f"preferences-v{version:04d}.json"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"version": version, "records": records}, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {"status": "published", "path": str(path), "count": len(records)}
