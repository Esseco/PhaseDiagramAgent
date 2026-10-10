"""Explicit serializable lifecycle fields; runtime dependencies are never saved."""

import json
from copy import deepcopy

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


def checkpoint_frame(frame):
    result = {}
    for key in FIELDS:
        if key in frame:
            value = frame[key]
            # Paths are protocol data; other runtime objects are errors, not silently dropped.
            from pathlib import Path

            encoded = json.dumps(
                value,
                default=lambda item: (
                    str(item)
                    if isinstance(item, Path)
                    else (_ for _ in ()).throw(TypeError(f"Nonserializable business field: {key}"))
                ),
            )
            result[key] = json.loads(encoded)
    return result


def restore_frame(runtime_frame, saved):
    return {**runtime_frame, **deepcopy(saved)}
