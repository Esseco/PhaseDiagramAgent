"""Bounded, thread-scoped conversation memory; never scientific state or approval."""

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from phase_agent.runtime.turn_process import clean

MAX_TURNS = 6
MAX_CONTEXT_CHARS = 10000


def memory_path(state_path, thread_id):
    key = hashlib.sha256(str(thread_id or "local").encode()).hexdigest()[:24]
    return Path(state_path).parent / "dialogue_memory" / (key + ".json")


def recent_turns(state_path, thread_id):
    try:
        rows = json.loads(memory_path(state_path, thread_id).read_text(encoding="utf-8"))
        selected = (
            [
                {
                    "user": str(row.get("user", ""))[:1600],
                    "assistant": str(row.get("assistant", ""))[:2400],
                    **(
                        {"facts_signature": row["facts_signature"]}
                        if row.get("facts_signature")
                        else {}
                    ),
                }
                for row in rows[-MAX_TURNS:]
            ]
            if isinstance(rows, list)
            else []
        )
        while (
            len(selected) > 1 and len(json.dumps(selected, ensure_ascii=False)) > MAX_CONTEXT_CHARS
        ):
            selected.pop(0)
        return selected
    except (OSError, ValueError, TypeError):
        return []


def remember_turn(state_path, thread_id, user, assistant, *, facts_signature=None):
    path = memory_path(state_path, thread_id)
    rows = recent_turns(state_path, thread_id)
    rows.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "user": str(clean(user))[:16000],
            "assistant": str(clean(assistant))[:16000],
            **({"facts_signature": facts_signature} if facts_signature else {}),
        }
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(rows[-MAX_TURNS:], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        pass
