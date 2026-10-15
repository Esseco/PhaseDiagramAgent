"""Atomic JSON helpers shared by the step runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def read_json(path, default=None):
    source = Path(path)
    if not source.is_file():
        return default
    if source.name == "state.json":
        from phase_agent.runtime.turn_process import reuse_initial_state

        found, value = reuse_initial_state(source)
        if found:
            from phase_agent.tools.state.workflow_control import restore_control

            return restore_control(source, value)
    if source.name != "state.json":
        value = json.loads(source.read_text(encoding="utf-8"))
        from phase_agent.tools.state.workflow_control import restore_control

        return restore_control(source, value)
    from phase_agent.runtime.turn_process import timed_call

    text = timed_call("json_read:" + source.name, source.read_text, encoding="utf-8", category="io")
    value = timed_call("json_decode:" + source.name, json.loads, text, category="io")
    from phase_agent.tools.state.workflow_control import restore_control

    return restore_control(source, value)


def write_json(path, payload):
    from phase_agent.tools.state.workflow_control import state_write_guard

    with state_write_guard(path, payload):
        return _write_json(path, payload)


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Large workflow states are machine records; whitespace alone can occupy
    # tens of MB. Keep other artifacts readable and preserve every state field.
    options = {"separators": (",", ":")} if target.name == "state.json" else {"indent": 2}
    from phase_agent.runtime.turn_process import timed_call

    encoded = (
        timed_call(
            "json_encode:" + target.name,
            json.dumps,
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            category="io",
            **options,
        )
        + "\n"
    ).encode("utf-8")
    if (
        target.is_file()
        and target.stat().st_size == len(encoded)
        and target.read_bytes() == encoded
    ):
        return str(target)
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    started = time.monotonic()
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(3):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
        from phase_agent.runtime.turn_process import record_timing

        record_timing("json_write:" + target.name, time.monotonic() - started, category="io")
    return str(target)


def content_id(payload, prefix="record"):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"
