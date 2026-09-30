"""Single state transition boundary for Agent-visible snapshots."""
from copy import deepcopy
import hashlib
import json

from analysis_layer.state.build_state_snapshot import build_state_snapshot
from config_layer.schema.action_state_schema import validate_state_schema
from data_layer.memory.decision_memory import update_short_term_memory
from data_layer.memory.collect_memory_candidates import collect_memory_candidates
from config_layer.schema.state_schema_migrations import migrate_persisted_state


def update_state_snapshot(state: dict, *, config_version=None) -> dict:
    current = collect_memory_candidates(migrate_persisted_state(state))
    fingerprint = _state_fingerprint(current)
    existing = current.get("current_state_snapshot") or {}
    if existing.get("source_fingerprint") == fingerprint:
        return current
    index = int(current.get("state_snapshot_index", 0)) + 1
    snapshot = build_state_snapshot(current, snapshot_index=index, config_version=config_version)
    errors = validate_state_schema(snapshot)
    if errors:
        raise ValueError("invalid state snapshot: " + ",".join(errors))
    snapshot["source_fingerprint"] = fingerprint
    current["state_snapshot_index"] = index
    current["current_state_snapshot"] = snapshot
    current.setdefault("state_snapshots", []).append(_history_snapshot(snapshot))
    current["state_snapshots"] = current["state_snapshots"][-50:]
    return update_short_term_memory(current, snapshot)


def agent_state_summary(state: dict) -> dict:
    return deepcopy(state.get("current_state_snapshot") or {})


def _history_snapshot(snapshot):
    """History indexes reference current ledger; old full snapshots stay readable."""
    return {"history_format": "summary-v1",
            **{key: deepcopy(snapshot.get(key)) for key in (
                "snapshot_id", "snapshot_index", "source_fingerprint", "config_version",
                "current_status", "available_budget", "budget_usage", "search_history")},
            "candidate_count": len(snapshot.get("available_branches") or []),
            "phase_versions": {method: item.get("version") for method, item in
                               (snapshot.get("current_convex_hull") or {}).items()},
            "memory_version": ((snapshot.get("decision_context") or {}).get("long_term_human_advice") or {}).get("version")}


def _state_fingerprint(state):
    excluded = {"current_state_snapshot", "state_snapshots", "state_snapshot_index"}
    payload = {key: value for key, value in state.items() if key not in excluded}
    memory = deepcopy(payload.get("decision_memory") or {})
    memory.pop("short_term", None)
    if "decision_memory" in payload:
        payload["decision_memory"] = memory
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
