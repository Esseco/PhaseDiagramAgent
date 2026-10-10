"""Keep phase recognition cache separate from runtime state, preserving legacy paths."""

from pathlib import Path


def phase_cache_location(state_path):
    source = Path(state_path)
    if source.parent.name in {"runtime", "workflow_state"}:
        return source.parent / "cache/phase_identification_cache.json"
    return source.with_name("phase_identification_cache.json")
