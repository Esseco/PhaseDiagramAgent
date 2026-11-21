"""Conservative reduction of an existing portable Relax+MC input inventory."""

import json
from copy import deepcopy
from pathlib import PurePosixPath


def reduced_payload(entries, batch, *, runtime_python_paths=None):
    """Deduplicate identical input bytes and omit desktop-only runtime files.

    Every identity-bearing record remains; duplicate paths are rebound only
    when their complete archived bytes have identical SHA256 and size.
    Other runtime modules and budget records are intentionally retained until
    their execution dependencies can be independently proven.
    """
    seen, aliases, kept = {}, {}, []
    for entry in entries:
        name = entry["path"]
        path = PurePosixPath(name)
        if name.startswith("runtime/phase_agent/") and (
            path.name == "README.md" or path.suffix in {".pyw", ".code-workspace"}
        ):
            continue
        if (
            runtime_python_paths is not None
            and name.startswith("runtime/phase_agent/")
            and path.suffix == ".py"
            and name.removeprefix("runtime/") not in runtime_python_paths
        ):
            continue
        if name.startswith("inputs/") and path.suffix == ".vasp":
            identity = (entry["sha256"], entry["size"])
            if identity in seen:
                aliases[name] = seen[identity]
                continue
            seen[identity] = name
        kept.append(deepcopy(entry))

    def remap(value):
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [remap(item) for item in value]
        return aliases.get(value, value) if isinstance(value, str) else value

    result = remap(batch)
    return kept, result, aliases
