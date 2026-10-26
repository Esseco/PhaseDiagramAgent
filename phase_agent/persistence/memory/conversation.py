"""Bounded, thread-scoped conversation memory; never scientific state or approval."""

from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime, timezone
from phase_agent.text_safety import clean

MAX_TURNS = 6
MAX_CONTEXT_CHARS = 10000


def memory_path(state_path, thread_id):
    key = hashlib.sha256(str(thread_id or "local").encode()).hexdigest()[:24]
    return Path(state_path).parent / "dialogue_memory" / (key + ".json")


def _stored_turns(state_path, thread_id):
    try:
        rows = json.loads(memory_path(state_path, thread_id).read_text(encoding="utf-8-sig"))
        return (
            [
                row
                for row in rows[-MAX_TURNS:]
                if isinstance(row, dict)
                and isinstance(row.get("user"), str)
                and isinstance(row.get("assistant"), str)
            ]
            if isinstance(rows, list)
            else []
        )
    except (OSError, ValueError, TypeError):
        return []


def recent_turns(state_path, thread_id):
    selected = [
        {
            "user": row["user"][:1600],
            "assistant": row["assistant"][:2400],
            **({"facts_signature": row["facts_signature"]} if row.get("facts_signature") else {}),
            **({"turn_id": row["turn_id"]} if row.get("turn_id") else {}),
        }
        for row in _stored_turns(state_path, thread_id)
    ]
    while len(selected) > 1 and len(json.dumps(selected, ensure_ascii=False)) > MAX_CONTEXT_CHARS:
        selected.pop(0)
    return selected


def remember_turn(
    state_path,
    thread_id,
    user,
    assistant,
    *,
    facts_signature=None,
    turn_id=None,
    replace_only=False,
):
    path = memory_path(state_path, thread_id)
    rows = _stored_turns(state_path, thread_id)
    index = next(
        (i for i, row in enumerate(rows) if turn_id and row.get("turn_id") == turn_id), None
    )
    if replace_only and index is None:
        return False
    previous = rows[index] if index is not None else {}
    row = {
        **previous,
        "time": previous.get("time") or datetime.now(timezone.utc).isoformat(),
        "user": str(clean(user))[:16000],
        "assistant": str(clean(assistant))[:16000],
        **({"facts_signature": facts_signature} if facts_signature else {}),
        **({"turn_id": turn_id} if turn_id else {}),
    }
    if index is None:
        rows.append(row)
    else:
        rows[index] = row
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from phase_agent.tools.step_runner.file_protocol import _write_json

        _write_json(path, rows[-MAX_TURNS:])
        return True
    except OSError as error:
        logging.getLogger(__name__).warning(
            "Conversation memory unavailable (%s); scientific state is unchanged",
            type(error).__name__,
        )
        return False
