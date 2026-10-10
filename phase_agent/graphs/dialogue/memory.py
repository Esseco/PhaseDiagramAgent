"""Bounded, thread-scoped conversation memory; never scientific state or approval."""

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from phase_agent.runtime.turn_process import clean


def memory_path(state_path, thread_id):
    key = hashlib.sha256(str(thread_id or "local").encode()).hexdigest()[:24]
    return Path(state_path).parent / "dialogue_memory" / (key + ".json")


def recent_turns(state_path, thread_id):
    try:
        rows = json.loads(memory_path(state_path, thread_id).read_text(encoding="utf-8"))
        return (
            [
                {
                    "user": str(row.get("user", ""))[:1000],
                    "assistant": str(row.get("assistant", ""))[:1600],
                }
                for row in rows[-6:]
            ]
            if isinstance(rows, list)
            else []
        )
    except (OSError, ValueError, TypeError):
        return []


def remember_turn(state_path, thread_id, user, assistant):
    path = memory_path(state_path, thread_id)
    rows = recent_turns(state_path, thread_id)
    rows.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "user": clean(user),
            "assistant": clean(assistant),
        }
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows[-12:], ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass
