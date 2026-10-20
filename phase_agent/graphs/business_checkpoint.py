"""Explicit serializable lifecycle fields; runtime dependencies are never saved."""

import json
from pathlib import Path

FIELDS = {
    "_workflow_prepared",
    "loaded_state",
    "automatic_results",
    "collection_report",
    "combined_results",
    "pre_reconciled",
    "training_returns",
    "training_handoffs",
    "feedback",
    "recovered_count",
    "manual_wait",
    "rebuilding",
    "recovery_question",
    "result",
    "effective_config",
    "snapshot",
    "runtime_state_path",
    "phase_cache_path",
}


def checkpoint_frame(frame, previous=None):
    """Detach JSON facts once, preserving shared state inside one snapshot."""
    memo, visiting = {}, set()
    previous = previous or {}
    return {
        key: _snapshot(value, previous.get(key), memo, visiting)
        for key, value in frame.items()
        if key in FIELDS
    }


def _snapshot(value, previous, memo, visiting):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (dict, list, tuple)):
        raise TypeError("Nonserializable business field: " + type(value).__name__)
    identity = id(value)
    if identity in visiting:
        raise ValueError("Circular reference detected")
    if identity in memo:
        return memo[identity]
    if value == previous:
        memo[identity] = previous
        return previous
    visiting.add(identity)
    try:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    if key is None or isinstance(key, (int, float, bool)):
                        key = json.dumps(key)
                    else:
                        raise TypeError("Checkpoint dictionary keys must be JSON keys")
                old = previous.get(key) if isinstance(previous, dict) else None
                result[key] = _snapshot(item, old, memo, visiting)
        else:
            # State histories are JSON arrays. Let the native codec detach
            # their trees rather than recursing through every scalar in Python.
            result = json.loads(json.dumps(value, default=_json_default))
        memo[identity] = result
        return result
    finally:
        visiting.remove(identity)


def restore_frame(runtime_frame, saved):
    from phase_agent.tools.state.workflow_control import FIELD

    current = runtime_frame.get("loaded_state") or {}
    previous = saved.get("loaded_state") or {}
    if current.get(FIELD) != previous.get(FIELD):
        raise ValueError("stale_lifecycle_checkpoint_control_revision")
    # Shared checkpoint snapshots are read-only. Restore independent mutable
    # stage facts, matching the historical JSON checkpoint ownership boundary.
    restored = {**runtime_frame, **json.loads(json.dumps(saved))}
    from phase_agent.tools.workflows.compact_action_history import compact_state_history

    # Older lifecycle checkpoints may still carry the redundant audit snapshots.
    # Apply the same migration after detaching, without editing saved checkpoints.
    if isinstance(restored.get("loaded_state"), dict):
        compact_state_history(restored["loaded_state"], _owned=True)
    for key in FIELDS - {"loaded_state"}:
        value = restored.get(key)
        if isinstance(value, dict) and isinstance(value.get("state"), dict):
            compact_state_history(value["state"], _owned=True)
    return restored


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Nonserializable business field: " + type(value).__name__)
