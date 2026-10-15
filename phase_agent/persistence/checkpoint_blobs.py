"""Deduplicate large business-state fields while preserving checkpoint values."""

from hashlib import sha256
import zlib

STATE_FIELDS = "__phase_state_fields_v1__"


class CheckpointBlobSerde:
    def __init__(self, delegate, connection, lock, threshold=65536):
        self.delegate, self.connection, self.lock = delegate, connection, lock
        self.raw_delegate = getattr(delegate, "delegate", delegate)
        self.threshold = threshold
        self.known = set()

    def _table(self):
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint_payloads (id TEXT PRIMARY KEY, kind TEXT NOT NULL, data BLOB NOT NULL)"
        )

    def _state(self, value, memo=None):
        if not isinstance(value, dict):
            return value
        memo = memo if memo is not None else {}
        cached = memo.get(("state", id(value)))
        if cached is not None and cached[0] is value:
            return cached[1]
        fields = {}
        for name, item in value.items():
            prior = memo.get(("field", id(item)))
            if prior is not None and prior[0] is item:
                fields[name] = prior[1]
                continue
            kind, data = self.raw_delegate.dumps_typed(item)
            if len(data) < self.threshold:
                fields[name] = {"inline": item}
                continue
            identity = sha256(kind.encode() + b"\0" + data).hexdigest()
            with self.lock:
                self._table()
                if identity not in self.known:
                    existing = self.connection.execute(
                        "SELECT 1 FROM checkpoint_payloads WHERE id=?", (identity,)
                    ).fetchone()
                    if existing is None:
                        compressed = zlib.compress(data, 1)
                        stored_kind, stored_data = (
                            ("zlib:" + kind, compressed)
                            if len(compressed) < len(data)
                            else (kind, data)
                        )
                        self.connection.execute(
                            "INSERT OR IGNORE INTO checkpoint_payloads VALUES (?,?,?)",
                            (identity, stored_kind, stored_data),
                        )
                    self.known.add(identity)
            fields[name] = {"blob": identity}
            memo[("field", id(item))] = (item, fields[name])
        result = {STATE_FIELDS: fields}
        memo[("state", id(value))] = (value, result)
        return result

    def _restore_state(self, value):
        if not isinstance(value, dict) or set(value) != {STATE_FIELDS}:
            return value
        result = {}
        for name, reference in value[STATE_FIELDS].items():
            if "inline" in reference:
                result[name] = reference["inline"]
            else:
                with self.lock:
                    row = self.connection.execute(
                        "SELECT kind,data FROM checkpoint_payloads WHERE id=?",
                        (reference["blob"],),
                    ).fetchone()
                if row is None:
                    raise ValueError(
                        "Checkpoint payload missing; recovery is blocked, not replayed"
                    )
                raw_kind, raw_data = (
                    (row[0][5:], zlib.decompress(row[1])) if row[0].startswith("zlib:") else row
                )
                if sha256(raw_kind.encode() + b"\0" + raw_data).hexdigest() == reference["blob"]:
                    result[name] = self.raw_delegate.loads_typed((raw_kind, raw_data))
                elif sha256(row[0].encode() + b"\0" + row[1]).hexdigest() == reference["blob"]:
                    result[name] = self.delegate.loads_typed(row)
                else:
                    raise ValueError("Checkpoint payload checksum mismatch")
        return result

    def _frame(self, value, restore=False, memo=None):
        if not isinstance(value, dict):
            return value
        state = self._restore_state if restore else lambda item: self._state(item, memo)
        result = dict(value)
        if isinstance(result.get("loaded_state"), dict):
            result["loaded_state"] = state(result["loaded_state"])
        for name in ("pre_reconciled", "feedback", "result"):
            if isinstance(result.get(name), dict) and isinstance(result[name].get("state"), dict):
                result[name] = {**result[name], "state": state(result[name]["state"])}
        return result

    def _value(self, value, restore=False, memo=None):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if isinstance(value.get("channel_values"), dict):
            channels = dict(value["channel_values"])
            if isinstance(channels.get("business_frame"), dict):
                channels["business_frame"] = self._frame(channels["business_frame"], restore, memo)
            if isinstance(channels.get("completed_response"), dict):
                channels["completed_response"] = self._value(
                    channels["completed_response"], restore, memo
                )
            result["channel_values"] = channels
        elif any(
            name in value for name in ("loaded_state", "pre_reconciled", "_workflow_prepared")
        ):
            result = self._frame(value, restore, memo)
        elif isinstance(value.get("state"), dict):
            state = self._restore_state if restore else lambda item: self._state(item, memo)
            result["state"] = state(value["state"])
        return result

    def dumps_typed(self, value):
        kind, data = self.delegate.dumps_typed(self._value(value, memo={}))
        return "phase-cas-v1:" + kind, data

    def loads_typed(self, value):
        kind, data = value
        if not kind.startswith("phase-cas-v1:"):
            return self.delegate.loads_typed(value)
        return self._value(
            self.delegate.loads_typed((kind[len("phase-cas-v1:") :], data)),
            restore=True,
        )
