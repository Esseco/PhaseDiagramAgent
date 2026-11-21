"""Bounded context-local read versions; no scientific data or global state cache."""

from contextvars import ContextVar
import hashlib
from pathlib import Path
from threading import Lock

_versions = ContextVar("phase_state_read_versions", default=None)
_strong_digest_lock = Lock()
_strong_digest_cache: dict[tuple[str, tuple], str] = {}
_LARGE_DIGEST_THRESHOLD = 1024 * 1024


def file_identity(path):
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def read_identity(path, *, strong=False):
    source = Path(path)
    control = source.with_name(source.name + ".control.json")
    before = (file_identity(source), file_identity(control))
    if not strong:
        return before

    def digest(path, ident):
        if ident is None:
            return None
        cache_key = (str(path.resolve()), ident) if ident[2] >= _LARGE_DIGEST_THRESHOLD else None
        if cache_key is not None:
            with _strong_digest_lock:
                cached = _strong_digest_cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            with path.open("rb") as stream:
                value = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
                from phase_agent.runtime.turn_process import record_read_metric

                record_read_metric(
                    "strong_file_validation", bytes_read=path.stat().st_size, path=str(path)
                )
                result = value.hexdigest()
                if cache_key is not None and file_identity(path) == ident:
                    with _strong_digest_lock:
                        while len(_strong_digest_cache) >= 16:
                            _strong_digest_cache.pop(next(iter(_strong_digest_cache)))
                        _strong_digest_cache[cache_key] = result
                return result
        except FileNotFoundError:
            return None

    contents = (digest(source, before[0]), digest(control, before[1]))
    if (file_identity(source), file_identity(control)) != before:
        raise ValueError("state_changed_during_identity")
    return (*before, contents)


def remember_read(path, identity):
    # Copy the small map so inherited async contexts never mutate each other.
    versions = dict(_versions.get() or {})
    key = str(Path(path).resolve())
    versions.pop(key, None)
    versions[key] = identity
    while len(versions) > 16:
        versions.pop(next(iter(versions)))
    _versions.set(versions)


def verify_read(path, identity):
    if read_identity(path) != identity:
        raise ValueError("state_changed_during_read; reload current state")
    remember_read(path, identity)


def check_write_version(path):
    key = str(Path(path).resolve())
    versions = _versions.get() or {}
    if key in versions and versions[key] != read_identity(path):
        raise ValueError("stale_state_file_version; reload state before saving")
