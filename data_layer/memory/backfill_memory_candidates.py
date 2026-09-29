"""Idempotently add factual memory candidates to an existing state file."""

import json
from pathlib import Path
import shutil

from data_layer.memory.collect_memory_candidates import collect_memory_candidates


def backfill_memory_candidates(state_path, *, dry_run=True):
    path = Path(state_path).resolve()
    original = json.loads(path.read_text(encoding="utf-8"))
    before = len(original.get("memory_candidates") or [])
    updated = collect_memory_candidates(original)
    after = len(updated.get("memory_candidates") or [])
    result = {"state_path": str(path), "before": before, "after": after,
              "added": after - before, "dry_run": bool(dry_run)}
    if dry_run or after == before:
        return result
    backup = path.with_name(path.name + ".memory-backup")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".memory-tmp")
    try:
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2,
                                        default=str) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result["backup_path"] = str(backup)
    return result
