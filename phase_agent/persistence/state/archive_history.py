"""Safely archive settled historical records from bloated state.json files."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import (
    read_json,
    write_json,
    write_json_if_version,
)

_ARCHIVE_FIELDS: tuple[tuple[str, str], ...] = (
    ("event_history", "keep_recent_events"),
    ("decisions", "keep_recent_decisions"),
    ("invocations", "keep_recent_invocations"),
    ("action_records", "keep_recent_actions"),
    ("cost_history", "keep_recent_costs"),
)

_ACTIVE_STATUSES = frozenset(
    {
        "pending",
        "awaiting_approval",
        "awaiting_human_input",
        "running",
        "submitted",
        "queued",
        "in_progress",
        "approved",
        "active",
    }
)

_ID_KEYS = ("decision_id", "action_id", "invocation_id", "policy_id", "proposal_id", "id")


def _collect_protected_ids(state: dict[str, Any]) -> set[str]:
    """Collect identifiers referenced by pending or active workflow objects."""
    protected: set[str] = set()

    def _scan(value: Any) -> None:
        if isinstance(value, dict):
            for key in _ID_KEYS:
                item = value.get(key)
                if isinstance(item, str) and item:
                    protected.add(item)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    _scan(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    _scan(child)

    for field in (
        "pending_execution_policies",
        "execution_reconciliations",
        "pending_proposal",
        "active_proposal",
        "last_decision",
        "latest_action",
    ):
        _scan(state.get(field))
    return protected


def _is_protected_record(record: Any, protected_ids: set[str]) -> bool:
    """Return True when a record is non-terminal or referenced by active state."""
    if not isinstance(record, dict):
        return False
    status = record.get("status")
    if isinstance(status, str) and status.lower() in _ACTIVE_STATUSES:
        return True
    for key in _ID_KEYS:
        identifier = record.get(key)
        if isinstance(identifier, str) and identifier in protected_ids:
            return True
    return False


def _partition_records(
    records: Any,
    keep_recent: int,
    protected_ids: set[str],
) -> tuple[list[Any], list[Any]]:
    """Split records into (archived, kept) while preserving relative order."""
    if not isinstance(records, list) or not records:
        return [], list(records) if isinstance(records, list) else []

    protected_flags = [_is_protected_record(row, protected_ids) for row in records]
    settled_indices = [idx for idx, flag in enumerate(protected_flags) if not flag]
    if keep_recent >= len(settled_indices):
        return [], list(records)

    keep_settled = set(settled_indices[-keep_recent:]) if keep_recent > 0 else set()
    archived: list[Any] = []
    kept: list[Any] = []
    for idx, row in enumerate(records):
        if protected_flags[idx] or idx in keep_settled:
            kept.append(row)
        else:
            archived.append(row)
    return archived, kept


def archive_state_history(
    state_path: str | Path,
    *,
    keep_recent_events: int = 200,
    keep_recent_decisions: int = 100,
    keep_recent_invocations: int = 100,
    keep_recent_actions: int = 200,
    keep_recent_costs: int = 200,
    destination_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Archive settled historical lists from state.json with SHA256 verification."""
    limits = {
        "keep_recent_events": keep_recent_events,
        "keep_recent_decisions": keep_recent_decisions,
        "keep_recent_invocations": keep_recent_invocations,
        "keep_recent_actions": keep_recent_actions,
        "keep_recent_costs": keep_recent_costs,
    }
    for name, value in limits.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    source = Path(state_path)
    if not source.is_file():
        raise FileNotFoundError(f"state file not found: {source}")

    bytes_before = source.stat().st_size
    identity = read_identity(source, strong=True)
    state = read_json(source)
    if not isinstance(state, dict):
        raise ValueError("invalid_state_payload")

    protected_ids = _collect_protected_ids(state)
    archived_records: dict[str, list[Any]] = {}
    kept_records: dict[str, list[Any]] = {}
    archived_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}

    for field, limit_key in _ARCHIVE_FIELDS:
        archived, kept = _partition_records(state.get(field), limits[limit_key], protected_ids)
        if archived:
            archived_records[field] = archived
        kept_records[field] = kept
        archived_counts[field] = len(archived)
        kept_counts[field] = len(kept)

    total_archived = sum(archived_counts.values())
    if total_archived == 0:
        return {
            "status": "noop",
            "state_path": str(source),
            "archive_path": None,
            "archived_counts": archived_counts,
            "kept_counts": kept_counts,
            "bytes_before": bytes_before,
            "bytes_after": bytes_before,
        }

    if dry_run:
        return {
            "status": "dry_run",
            "state_path": str(source),
            "archive_path": None,
            "archived_counts": archived_counts,
            "kept_counts": kept_counts,
            "bytes_before": bytes_before,
            "bytes_after": bytes_before,
        }

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir = (
        Path(destination_dir) if destination_dir is not None else source.parent / "history"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"state_history_{timestamp}.json"

    archive_payload = {
        "schema_version": 1,
        "archived_at": now.isoformat(),
        "source_state_path": str(source),
        "source_identity": list(identity) if isinstance(identity, tuple) else identity,
        "archived_counts": archived_counts,
        "records": archived_records,
    }
    write_json(archive_path, archive_payload)

    raw_archive = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(raw_archive).hexdigest()
    verified_payload = json.loads(raw_archive.decode("utf-8"))
    for field, count in archived_counts.items():
        actual = len((verified_payload.get("records") or {}).get(field) or [])
        if actual != count:
            raise ValueError(f"archive_verification_failed:{field}:{actual}!={count}")

    slimmed_state = deepcopy(state)
    for field, _ in _ARCHIVE_FIELDS:
        if field in slimmed_state or kept_records[field]:
            slimmed_state[field] = kept_records[field]
    slimmed_state.setdefault("history_archives", []).append(
        {
            "archive_path": str(archive_path),
            "sha256": archive_sha256,
            "archived_at": now.isoformat(),
            "archived_counts": archived_counts,
        }
    )

    write_json_if_version(source, slimmed_state, identity)
    bytes_after = source.stat().st_size

    return {
        "status": "archived",
        "state_path": str(source),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archived_counts": archived_counts,
        "kept_counts": kept_counts,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive old settled history from state.json.")
    parser.add_argument("--state", required=True, help="Path to workflow_state/state.json")
    parser.add_argument("--destination", default=None, help="Optional archive output directory")
    parser.add_argument("--keep-events", type=int, default=200)
    parser.add_argument("--keep-decisions", type=int, default=100)
    parser.add_argument("--keep-invocations", type=int, default=100)
    parser.add_argument("--keep-actions", type=int, default=200)
    parser.add_argument("--keep-costs", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = archive_state_history(
        args.state,
        keep_recent_events=args.keep_events,
        keep_recent_decisions=args.keep_decisions,
        keep_recent_invocations=args.keep_invocations,
        keep_recent_actions=args.keep_actions,
        keep_recent_costs=args.keep_costs,
        destination_dir=args.destination,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
