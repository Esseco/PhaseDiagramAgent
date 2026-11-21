"""Explicit serializable lifecycle fields; runtime dependencies are never saved."""

import json
from pathlib import Path

from phase_agent.persistence.checkpoint_blobs import SealedCheckpointList

FIELDS = {
    "_workflow_prepared",
    "loaded_state",
    "automatic_results",
    "collection_report",
    "combined_results",
    "pre_reconciled",
    "training_returns",
    "relax_recovery_updated",
    "idle_exact_feedback_reused",
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


def checkpoint_frame(frame, previous=None, *, seal_lists=False):
    """Detach JSON facts once, preserving shared state inside one snapshot."""
    memo, visiting = {}, set()
    previous = previous or {}
    return {
        key: _snapshot(
            value,
            previous.get(key),
            memo,
            visiting,
            layout=(
                "state"
                if key == "loaded_state"
                else "container"
                if key in {"pre_reconciled", "feedback", "result"}
                else None
            )
            if seal_lists
            else None,
        )
        for key, value in frame.items()
        if key in FIELDS
    }


def _snapshot(value, previous, memo, visiting, *, layout=None):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (dict, list, tuple)):
        raise TypeError("Nonserializable business field: " + type(value).__name__)
    identity = id(value)
    if identity in visiting:
        raise ValueError("Circular reference detected")
    key_in_memo = (identity, layout)
    if key_in_memo in memo:
        return memo[key_in_memo]
    if value == previous:
        memo[key_in_memo] = previous
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
                child_layout = (
                    "state"
                    if layout == "container" and key == "state"
                    else "field"
                    if layout == "state" and type(item) is list and len(item) >= 1024
                    else None
                )
                result[key] = _snapshot(item, old, memo, visiting, layout=child_layout)
        else:
            try:
                result = _detach_json_tree(value)
            except (TypeError, RecursionError):
                result = json.loads(json.dumps(value, default=_json_default))
            if layout == "field":
                result = SealedCheckpointList(result)
        memo[key_in_memo] = result
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
    try:
        detached = _detach_json_tree(saved)
    except (TypeError, RecursionError):
        detached = json.loads(json.dumps(saved, default=_json_default))
    return {**runtime_frame, **detached}


def _detach_json_tree(value):
    if isinstance(value, SealedCheckpointList):
        return value.mutable_copy()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        import math

        if math.isfinite(value):
            return value
        raise TypeError("nonfinite JSON value")
    if type(value) is list:
        return [_detach_json_tree(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _detach_json_tree(item) for key, item in value.items()}
    raise TypeError("nonstandard JSON value")


def _json_default(value):
    if isinstance(value, SealedCheckpointList):
        return value.mutable_copy()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Nonserializable business field: " + type(value).__name__)
