"""Deduplicate large business-state fields while preserving checkpoint values."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from weakref import ref
import zlib

STATE_FIELDS = "__phase_state_fields_v1__"
STATE_FIELDS_V2 = "__phase_state_fields_v2__"


class SealedCheckpointList:
    """Own a detached checkpoint list with no mutable access from graph nodes."""

    __slots__ = ("__rows", "__weakref__")

    def __init__(self, detached_rows):
        object.__setattr__(self, "_SealedCheckpointList__rows", detached_rows)

    def __setattr__(self, name, value):
        raise AttributeError("sealed checkpoint list")

    def __eq__(self, other):
        if isinstance(other, SealedCheckpointList):
            other = other.__rows
        return self.__rows == other

    def __deepcopy__(self, memo):
        return self

    def mutable_copy(self):
        return deepcopy(self.__rows)

    def _stream(self, serde):
        return serde._stream_list(self.__rows)

    def _inline_for_serialization(self):
        return self.__rows


class CheckpointBlobSerde:
    def __init__(self, delegate, connection, lock, threshold=65536):
        self.delegate, self.connection, self.lock = delegate, connection, lock
        self.raw_delegate = getattr(delegate, "delegate", delegate)
        self.threshold = threshold
        self._sealed_lists = {}
        self._routing_only = ContextVar("checkpoint_routing_only", default=False)
        self._wait_checkpoint_id = ContextVar("wait_checkpoint_id", default=None)

    @contextmanager
    def share_wait_history(self, checkpoint_id):
        """Share sealed lists in the exact checkpoint discarded by native wait."""
        token = self._wait_checkpoint_id.set(checkpoint_id)
        try:
            yield
        finally:
            self._wait_checkpoint_id.reset(token)

    @contextmanager
    def routing_only(self):
        """Inspect graph coordinates without materializing business state."""
        token = self._routing_only.set(True)
        try:
            yield
        finally:
            self._routing_only.reset(token)

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
            if isinstance(item, SealedCheckpointList):
                identity = self._sealed_list_reference(item)
                if identity is not None:
                    fields[name] = {"blob": identity}
                    memo[("field", id(item))] = (item, fields[name])
                    continue
                item = item.mutable_copy()
            elif type(item) is list and len(item) >= 1024:
                identity = self._stream_list(item)
                if identity is not None:
                    fields[name] = {"blob": identity}
                    memo[("field", id(item))] = (item, fields[name])
                    continue
            kind, data = self.raw_delegate.dumps_typed(item)
            if len(data) < self.threshold:
                fields[name] = {"inline": item}
                continue
            identity = sha256(kind.encode() + b"\0" + data).hexdigest()
            with self.lock:
                self._table()
                # A previously seen payload may have disappeared after a transaction rollback.
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
            fields[name] = {"blob": identity}
            memo[("field", id(item))] = (item, fields[name])
        instance = memo.get("next_state_instance", 0)
        memo["next_state_instance"] = instance + 1
        # Optional metadata: older readers ignore it. Preserve only actual
        # shared state identities, never alias independently equal states.
        result = {
            STATE_FIELDS: {
                name: {**reference, "state_instance": instance}
                for name, reference in fields.items()
            }
        }
        memo[("state", id(value))] = (value, result)
        return result

    def _sealed_list_reference(self, item):
        key = id(item)
        with self.lock:
            prior = self._sealed_lists.get(key)
            if prior is not None and prior[0]() is item:
                # The write referencing this row must share the DB write transaction.
                started = not self.connection.in_transaction
                if started:
                    self.connection.execute("BEGIN IMMEDIATE")
                try:
                    exists = self.connection.execute(
                        "SELECT 1 FROM checkpoint_payloads WHERE id=?", (prior[1],)
                    ).fetchone()
                    if exists is not None:
                        return prior[1]
                except BaseException:
                    if started:
                        self.connection.rollback()
                    raise
                if started:
                    self.connection.rollback()
        identity = item._stream(self)
        if identity is not None:
            with self.lock:
                if len(self._sealed_lists) >= 16:
                    self._sealed_lists.pop(next(iter(self._sealed_lists)))
                self._sealed_lists[key] = (ref(item), identity)
        return identity

    def _stream_list(self, value):
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        import struct
        import tempfile

        if type(self.raw_delegate) is not JsonPlusSerializer or not hasattr(
            self.connection, "blobopen"
        ):
            return None
        size = len(value)
        if size >= 2**32:
            return None
        header = (
            bytes([0x90 + size])
            if size < 16
            else b"\xdc" + struct.pack(">H", size)
            if size < 65536
            else b"\xdd" + struct.pack(">I", size)
        )
        digest = sha256(b"msgpack\0")
        raw_size = 0
        with tempfile.TemporaryFile() as raw, tempfile.TemporaryFile() as compressed:

            def append(part):
                nonlocal raw_size
                raw_size += len(part)
                digest.update(part)
                raw.write(part)

            append(header)
            for offset in range(0, size, 256):
                part = value[offset : offset + 256]
                kind, data = self.raw_delegate.dumps_typed(part)
                if kind != "msgpack" or not data:
                    return None
                first = data[0]
                head = (
                    1
                    if 0x90 <= first <= 0x9F
                    else 3
                    if first == 0xDC
                    else 5
                    if first == 0xDD
                    else 0
                )
                if not head:
                    return None
                count = first - 0x90 if head == 1 else int.from_bytes(data[1:head], "big")
                if count != len(part):
                    return None
                append(data[head:])
            if raw_size < self.threshold:
                return None
            identity = digest.hexdigest()
            with self.lock:
                self._table()
                existing = self.connection.execute(
                    "SELECT 1 FROM checkpoint_payloads WHERE id=?", (identity,)
                ).fetchone()
            if existing is not None:
                return identity
            # Content has been encoded and hashed, not inferred from mutable identity.
            # Compress only misses; the transactional write below still rechecks.
            compressor = zlib.compressobj(1)
            raw.seek(0)
            while part := raw.read(1024 * 1024):
                compressed.write(compressor.compress(part))
            compressed.write(compressor.flush())
            compressed_size = compressed.tell()
            source, stored_kind, stored_size = (
                (compressed, "zlib:msgpack", compressed_size)
                if compressed_size < raw_size
                else (raw, "msgpack", raw_size)
            )
            with self.lock:
                self._table()
                started = not self.connection.in_transaction
                if started:
                    self.connection.execute("BEGIN IMMEDIATE")
                try:
                    # Recheck storage even after a prior write: its transaction may have rolled back.
                    existing = self.connection.execute(
                        "SELECT 1 FROM checkpoint_payloads WHERE id=?", (identity,)
                    ).fetchone()
                    if existing is None:
                        self.connection.execute("SAVEPOINT phase_stream_blob")
                        try:
                            self.connection.execute(
                                "INSERT INTO checkpoint_payloads VALUES (?,?,zeroblob(?))",
                                (identity, stored_kind, stored_size),
                            )
                            rowid = self.connection.execute(
                                "SELECT rowid FROM checkpoint_payloads WHERE id=?", (identity,)
                            ).fetchone()[0]
                            source.seek(0)
                            with self.connection.blobopen(
                                "checkpoint_payloads", "data", rowid
                            ) as blob:
                                while part := source.read(1024 * 1024):
                                    blob.write(part)
                            self.connection.execute("RELEASE SAVEPOINT phase_stream_blob")
                        except BaseException:
                            self.connection.execute("ROLLBACK TO SAVEPOINT phase_stream_blob")
                            self.connection.execute("RELEASE SAVEPOINT phase_stream_blob")
                            raise
                except BaseException:
                    if started:
                        self.connection.rollback()
                    raise
            return identity

    def _restore_state(self, value, memo=None):
        if not isinstance(value, dict):
            return value
        marker = next(
            (name for name in (STATE_FIELDS, STATE_FIELDS_V2) if set(value) == {name}), None
        )
        if marker is None:
            return value
        if self._routing_only.get():
            return {}
        memo = memo if memo is not None else {}
        fields = value[marker]
        identities = {ref.get("state_instance") for ref in fields.values()}
        if len(identities) > 1 and any(item is not None for item in identities):
            raise ValueError("Checkpoint state alias mismatch")
        instance = next(iter(identities)) if len(identities) == 1 else None
        if instance is not None and (type(instance) is not int or instance < 0):
            raise ValueError("Invalid checkpoint state alias")
        key = ("restored_state", instance)
        if instance is not None and key in memo:
            prior_fields, prior_state = memo[key]
            if prior_fields != fields:
                raise ValueError("Checkpoint state alias mismatch")
            return prior_state
        result = {}
        for name, reference in fields.items():
            if "inline" in reference:
                result[name] = reference["inline"]
            else:
                chunks = reference.get("chunks")
                identities = chunks if chunks is not None else [reference["blob"]]
                parts = []
                share_list = chunks is None and memo.get("share_wait_history")
                for identity in identities:
                    blob_key = ("wait_history_blob", identity)
                    if share_list and blob_key in memo:
                        result[name] = memo[blob_key]
                        continue
                    with self.lock:
                        row = self.connection.execute(
                            "SELECT kind,data FROM checkpoint_payloads WHERE id=?", (identity,)
                        ).fetchone()
                    if row is None:
                        raise ValueError(
                            "Checkpoint payload missing; recovery is blocked, not replayed"
                        )
                    raw_kind, raw_data = (
                        (row[0][5:], zlib.decompress(row[1])) if row[0].startswith("zlib:") else row
                    )
                    if sha256(raw_kind.encode() + b"\0" + raw_data).hexdigest() == identity:
                        item = self.raw_delegate.loads_typed((raw_kind, raw_data))
                    elif sha256(row[0].encode() + b"\0" + row[1]).hexdigest() == identity:
                        item = self.delegate.loads_typed(row)
                    else:
                        raise ValueError("Checkpoint payload checksum mismatch")
                    if chunks is not None:
                        if not isinstance(item, list):
                            raise ValueError("Invalid checkpoint chunk type")
                        parts.extend(item)
                    else:
                        if share_list and type(item) is list and len(item) >= 1024:
                            item = SealedCheckpointList(item)
                            memo[blob_key] = item
                        result[name] = item
                if chunks is not None:
                    if (
                        type(reference.get("length")) is not int
                        or len(parts) != reference["length"]
                    ):
                        raise ValueError("Checkpoint chunk length mismatch")
                    result[name] = parts
        if instance is not None:
            memo[key] = (fields, result)
        return result

    def _frame(self, value, restore=False, memo=None, *, inline=False):
        if not isinstance(value, dict):
            return value
        if inline:

            def state(item):
                return {
                    key: field._inline_for_serialization()
                    if isinstance(field, SealedCheckpointList)
                    else field
                    for key, field in item.items()
                }
        else:
            state = (
                (lambda item: self._restore_state(item, memo))
                if restore
                else lambda item: self._state(item, memo)
            )
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
        memo = memo if memo is not None else {}
        result = dict(value)
        if not restore and isinstance(value.get("business_frame"), dict):
            result["business_frame"] = self._frame(value["business_frame"], inline=True)
        if isinstance(value.get("channel_values"), dict):
            if restore and value.get("id") and value["id"] == self._wait_checkpoint_id.get():
                memo["share_wait_history"] = True
            channels = dict(value["channel_values"])
            # LangGraph also persists the child's input on __start__. Preserve
            # the historical inline representation on this internal channel.
            start = channels.get("__start__")
            if (
                not restore
                and isinstance(start, dict)
                and isinstance(start.get("business_frame"), dict)
            ):
                channels["__start__"] = {
                    **start,
                    "business_frame": self._frame(start["business_frame"], inline=True),
                }
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
            state = (
                (lambda item: self._restore_state(item, memo))
                if restore
                else lambda item: self._state(item, memo)
            )
            result["state"] = state(value["state"])
        return result

    def dumps_typed(self, value):
        return next(self.dumps_many((value,)))

    def dumps_many(self, values):
        """Reuse identity work only within one immutable pending-write batch."""
        memo = {}
        for value in values:
            kind, data = self.delegate.dumps_typed(self._value(value, memo=memo))
            yield "phase-cas-v1:" + kind, data

    def loads_typed(self, value):
        kind, data = value
        prefix = next((p for p in ("phase-cas-v1:", "phase-cas-v2:") if kind.startswith(p)), None)
        if prefix is None:
            return self.delegate.loads_typed(value)
        return self._value(self.delegate.loads_typed((kind[len(prefix) :], data)), restore=True)
