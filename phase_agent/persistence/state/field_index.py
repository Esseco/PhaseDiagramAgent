"""Disposable, version-bound offsets into authoritative state.json."""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from phase_agent.persistence.state.read_versions import read_identity, verify_read


class IndexUnavailable(BaseException):
    """Force the outer read boundary to reload, even across broad exception handlers."""


def _identity_payload(identity):
    return json.loads(json.dumps(identity))


def _digest(fields):
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _control_digest(path):
    source = Path(path)
    control = source.with_name(source.name + ".control.json")
    try:
        with control.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except FileNotFoundError:
        return None


def publish_index(path, fields, expected_identity):
    from phase_agent.tools.step_runner.file_protocol import _write_json

    path = Path(path)
    before = read_identity(path)
    if before != expected_identity:
        raise IndexUnavailable("state changed before index publication")
    control_digest = _control_digest(path)
    record = {
        "version": 2,
        "source": _identity_payload(before),
        "fields": fields,
        "control_digest": control_digest,
        "digest": _digest({"fields": fields, "control_digest": control_digest}),
    }
    if read_identity(path) != before:
        raise IndexUnavailable("state changed before index publication")
    _write_json(path.with_name(path.name + ".index.json"), record)


class IndexedState(Mapping):
    def __init__(self, path, record, identity):
        self.path = Path(path)
        self.fields = record["fields"]
        self.manifest_digest = record.get("digest")
        self.identity = identity
        self.control_digest = record.get("control_digest")
        self.cache = {}

    @property
    def __class__(self):
        # A dict-specific check must fall back instead of silently selecting another branch.
        raise IndexUnavailable("unsupported state type check")

    def __str__(self):
        raise IndexUnavailable("unsupported state serialization")

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self.items()), memo)

    def verify(self, *, remember=False, strong=False):
        try:
            if read_identity(self.path) != self.identity:
                raise IndexUnavailable("state changed during indexed read")
            if strong or remember:
                self._verify_content()
            if remember:
                verify_read(self.path, self.identity)
        except (OSError, ValueError) as error:
            raise IndexUnavailable("state identity could not be verified") from error

    def _verify_content(self):
        """Validate keys and field bytes once at the read boundary, in bounded memory."""
        with self.path.open("rb") as stream:
            if stream.read(1) != b"{":
                raise IndexUnavailable("indexed state structure changed")
            for number, key in enumerate(sorted(self.fields)):
                offset, length, expected = self.fields[key]
                prefix = (
                    (b"," if number else b"")
                    + json.dumps(key, ensure_ascii=False).encode("utf-8")
                    + b":"
                )
                if stream.tell() + len(prefix) != offset or stream.read(len(prefix)) != prefix:
                    raise IndexUnavailable("indexed state fields changed")
                digest = hashlib.sha256()
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise IndexUnavailable("indexed state field truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if digest.hexdigest() != expected:
                    raise IndexUnavailable("indexed state field checksum mismatch")
            if stream.read(3) != b"}\n":
                raise IndexUnavailable("indexed state suffix changed")
        from phase_agent.runtime.turn_process import record_read_metric

        record_read_metric("index_full_validation", bytes_read=self.path.stat().st_size)
        if _control_digest(self.path) != self.control_digest:
            raise IndexUnavailable("control changed during indexed verification")
        if read_identity(self.path) != self.identity:
            raise IndexUnavailable("state changed during indexed verification")

    def __getitem__(self, key):
        self.verify()
        if key in self.cache:
            from phase_agent.runtime.turn_process import record_read_metric

            record_read_metric("field_cache_hit", field=key)
            return self.cache[key]
        offset, length, expected = self.fields[key]
        try:
            with self.path.open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(length)
        except OSError as error:
            raise IndexUnavailable("state field could not be read") from error
        self.verify()
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != expected:
            raise IndexUnavailable("state field checksum mismatch")
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise IndexUnavailable("invalid indexed JSON") from error
        from phase_agent.runtime.turn_process import record_read_metric

        record_read_metric("field_read", bytes_read=len(raw), field=key)
        self.cache[key] = value
        return value


_verified_control_header = None


def _control_header(path, digest):
    """Read only the authority fields that determine index visibility."""
    from phase_agent.tools.state.workflow_control import read_control

    global _verified_control_header
    key = (path.resolve(), digest)
    cached = _verified_control_header
    if cached and cached[0] == key:
        if _control_digest(path) != digest:
            raise IndexUnavailable("control content changed")
        return cached[1]
    control = read_control(path, expected_digest=digest)
    header = (
        (
            control["revision"],
            bool(control.get("path_relocations")),
            control.get("path_relocation_revision"),
        )
        if control
        else None
    )
    _verified_control_header = key, header
    return header


def open_indexed_state(path):
    from phase_agent.runtime.turn_process import record_read_metric

    def unavailable(reason):
        record_read_metric("index_fallback", reason=reason)
        return None

    path = Path(path)
    try:
        before = read_identity(path)
        record = json.loads(path.with_name(path.name + ".index.json").read_text(encoding="utf-8"))
        fields = record["fields"]
        version = record.get("version")
        if record.get("source") != _identity_payload(before) or not isinstance(fields, dict):
            return unavailable("source_or_field_manifest_mismatch")
        if version == 1:
            # Old indexes never bound control contents; only an absent control is safe.
            if before[1] is not None or record.get("digest") != _digest(fields):
                return unavailable("index_digest_or_control_mismatch")
        elif version == 2:
            control_digest = record.get("control_digest")
            if (
                "control_digest" not in record
                or record.get("digest")
                != _digest({"fields": fields, "control_digest": control_digest})
            ):
                return unavailable("index_digest_or_control_mismatch")
        else:
            return unavailable("index_version_offsets_or_control_revision_mismatch")
        size = before[0][2] if before[0] else 0
        if any(
            not isinstance(key, str)
            or not isinstance(value, list)
            or len(value) != 3
            or type(value[0]) is not int
            or type(value[1]) is not int
            or value[0] < 0
            or value[1] < 0
            or value[0] + value[1] > size
            or not isinstance(value[2], str)
            or len(value[2]) != 64
            for key, value in fields.items()
        ):
            return unavailable("index_version_offsets_or_control_revision_mismatch")
        if read_identity(path) != before:
            return unavailable("index_version_offsets_or_control_revision_mismatch")
        state = IndexedState(path, record, before)
        from phase_agent.tools.state.workflow_control import FIELD

        header = _control_header(path, state.control_digest)
        if header is not None:
            revision, has_relocations, relocation_revision = header
            if state.get(FIELD) != revision or (
                has_relocations and state.get("path_relocation_revision") != relocation_revision
            ):
                return unavailable("index_version_offsets_or_control_revision_mismatch")
        state.verify()
        record_read_metric("index_hit")
        return state
    except (OSError, ValueError, TypeError, KeyError, IndexUnavailable) as error:
        return unavailable(type(error).__name__ + ":" + str(error)[:160])


class PendingState(Mapping):
    """Read-only state overlay that hides the internal search-mode proposal."""

    def __init__(self, state):
        self.state = state
        self.pending = dict(state.get("pending_execution_policies") or {})
        self.pending.pop("__search_mode__", None)

    @property
    def __class__(self):
        raise IndexUnavailable("unsupported state type check")

    def __str__(self):
        raise IndexUnavailable("unsupported state serialization")

    def __getitem__(self, key):
        if key == "pending_execution_policies":
            return self.pending
        return self.state[key]

    def __iter__(self):
        yield from self.state
        if "pending_execution_policies" not in self.state:
            yield "pending_execution_policies"

    def __len__(self):
        return len(self.state) + ("pending_execution_policies" not in self.state)
