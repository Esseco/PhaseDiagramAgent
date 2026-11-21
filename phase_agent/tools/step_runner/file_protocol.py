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
    if source.name == "state.json":
        from phase_agent.persistence.state.read_versions import read_identity, verify_read

        identity = read_identity(source)
    if not source.is_file():
        if source.name == "state.json":
            verify_read(source, identity)
        return default
    if source.name == "state.json":
        from phase_agent.runtime.turn_process import reuse_initial_state

        found, value = reuse_initial_state(source)
        if found:
            from phase_agent.tools.state.workflow_control import restore_control

            value = restore_control(source, value)
            verify_read(source, identity)
            return value
    if source.name == "state.json":
        from phase_agent.runtime.turn_process import record_read_metric

        record_read_metric("full_state_parse", bytes_read=source.stat().st_size)
    if source.name != "state.json":
        value = json.loads(source.read_text(encoding="utf-8"))
        from phase_agent.tools.state.workflow_control import restore_control

        return restore_control(source, value)
    from phase_agent.runtime.turn_process import timed_call

    text = timed_call(
        "json_read:" + source.name, source.read_text, encoding="utf-8-sig", category="io"
    )
    value = timed_call("json_decode:" + source.name, json.loads, text, category="io")
    from phase_agent.tools.state.workflow_control import restore_control

    value = restore_control(source, value)
    verify_read(source, identity)
    return value


def write_json(path, payload):
    from phase_agent.tools.state.workflow_control import state_write_guard

    with state_write_guard(path, payload):
        return _write_json(path, payload)


def write_json_if_version(path, payload, expected_identity):
    """Publish a state read by this operation only if its source is unchanged."""
    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.tools.state.workflow_control import state_write_guard

    with state_write_guard(path, payload):
        strong = isinstance(expected_identity, (tuple, list)) and len(expected_identity) == 3
        if read_identity(path, strong=strong) != expected_identity:
            raise ValueError("stale_state_file_version; reload state before saving")
        return _write_json(path, payload)


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name == "state.json":
        changed, fields, source_identity = _write_state_stream(target, payload)
        if fields is not None:
            from phase_agent.persistence.state.field_index import IndexUnavailable, publish_index

            try:
                publish_index(target, fields, source_identity)
            except (OSError, TypeError, ValueError, IndexUnavailable):
                pass  # The index is disposable and cannot invalidate saved state.
        if changed:
            try:
                from phase_agent.persistence.state.state_view import publish_state_view

                publish_state_view(target, payload, expected_identity=source_identity)
            except (OSError, TypeError, ValueError):
                pass
        return str(target)
    options = {"indent": 2}
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


def _write_state_stream(target, payload):
    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    started = time.monotonic()
    try:
        fields = None
        with temporary.open("wb") as stream:
            if type(payload) is dict and all(type(key) is str for key in payload):
                fields = {}
                stream.write(b"{")
                for number, key in enumerate(sorted(payload)):
                    if number:
                        stream.write(b",")
                    stream.write(json.dumps(key, ensure_ascii=False).encode("utf-8") + b":")
                    offset = stream.tell()
                    raw = json.dumps(
                        payload[key],
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    stream.write(raw)
                    fields[key] = [
                        offset,
                        stream.tell() - offset,
                        hashlib.sha256(raw).hexdigest(),
                    ]
                stream.write(b"}\n")
            else:
                import io

                wrapper = io.TextIOWrapper(
                    stream, encoding="utf-8", newline="\n", write_through=True
                )
                json.dump(
                    payload,
                    wrapper,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
                wrapper.write("\n")
                wrapper.flush()
                wrapper.detach()
            stream.flush()
            os.fsync(stream.fileno())
        from phase_agent.persistence.state.read_versions import file_identity, read_identity

        temporary_identity = file_identity(temporary)
        before = read_identity(target)
        if target.is_file() and target.stat().st_size == temporary.stat().st_size:
            with target.open("rb") as old, temporary.open("rb") as new:
                while True:
                    left, right = old.read(65536), new.read(65536)
                    if left != right:
                        break
                    if not left:
                        after = read_identity(target)
                        if after != before:
                            raise ValueError("state_changed_during_write")
                        return False, fields, after
        for attempt in range(3):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))
        source_identity = read_identity(target)
        if (
            source_identity[0] is None
            or temporary_identity is None
            or source_identity[0][:4] != temporary_identity[:4]
        ):
            raise ValueError("state_changed_during_write")
        return True, fields, source_identity
    finally:
        temporary.unlink(missing_ok=True)
        from phase_agent.runtime.turn_process import record_timing

        record_timing("json_write:" + target.name, time.monotonic() - started, category="io")


def content_id(payload, prefix="record"):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"
