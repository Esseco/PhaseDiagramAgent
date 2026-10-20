"""Durable incremental phase cache; compact to the legacy JSON after a pass."""

import json
import os
from pathlib import Path

from filelock import FileLock

REUSABLE = {"identified", "outside_boundary"}


def journal_path(path):
    path = Path(path)
    return path.with_name(path.name + ".journal.jsonl")


def _read(cache, path):
    warnings = []
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("phase cache must be an object")
            cache.update(
                {
                    k: v
                    for k, v in stored.items()
                    if isinstance(v, dict) and v.get("status") in REUSABLE
                }
            )
        except (OSError, ValueError) as error:
            warnings.append(f"{type(error).__name__}: {error}")
    journal = journal_path(path)
    if journal.is_file():
        try:
            with journal.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or len(row) != 1:
                            raise ValueError("phase journal entry must contain one result")
                        cache.update(
                            {
                                k: v
                                for k, v in row.items()
                                if isinstance(v, dict) and v.get("status") in REUSABLE
                            }
                        )
                    except ValueError:
                        if len(warnings) < 8:
                            warnings.append(
                                f"phase cache journal line {number} is incomplete or invalid"
                            )
        except OSError as error:
            warnings.append(f"{type(error).__name__}: {error}")
    return "; ".join(warnings) if warnings else None


def load_cache(cache, path):
    if path is None:
        return None
    path = Path(path)
    if not path.parent.is_dir():
        return None
    with FileLock(str(path) + ".lock", timeout=10):
        return _read(cache, path)


def append_entry(path, key, evidence):
    if path is None:
        return
    if not isinstance(evidence, dict) or evidence.get("status") not in REUSABLE:
        raise ValueError("only successful phase identifications may be cached")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps({key: evidence}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with FileLock(str(path) + ".lock", timeout=10):
        with journal_path(path).open("a+b") as stream:
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def compact_cache(path, cache):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        merged = {}
        _read(merged, path)
        merged.update(
            {k: v for k, v in cache.items() if isinstance(v, dict) and v.get("status") in REUSABLE}
        )
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        # Retire the journal only after the merged main cache is durable. Both
        # operations share the writer lock, including readers of both files.
        with journal_path(path).open("wb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        cache.clear()
        cache.update(merged)
