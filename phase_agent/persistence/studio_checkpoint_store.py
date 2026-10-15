"""Lazy, incremental storage for Studio's nested checkpoint dictionaries.

The server keeps its existing saver and serialization contract. Only its three
PersistentDict containers change; scientific state and approvals are unrelated.
"""

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from hashlib import sha256
import pickle
import sqlite3
import threading
import weakref
import zlib

_databases = weakref.WeakValueDictionary()
_registry_lock = threading.RLock()
_MISSING = object()


class MigrationRequired(RuntimeError):
    pass


def _encode(value):
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


class _Database:
    def __init__(self, path, cache_bytes=8 * 1024**2):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-4096")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                path BLOB PRIMARY KEY, parent BLOB NOT NULL, key BLOB NOT NULL,
                kind TEXT NOT NULL, value BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS node_parent ON nodes(parent);
            CREATE TABLE IF NOT EXISTS payloads (id TEXT PRIMARY KEY, data BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.max_cache_bytes = cache_bytes

    def invalidate(self):
        self.cache.clear()
        self.cache_bytes = 0

    def value(self, path, data):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path][0]
        raw = zlib.decompress(data[1:]) if data[:1] == b"Z" else data[1:]
        if isinstance(path, str) and sha256(raw).hexdigest() != path:
            raise ValueError("Studio history payload checksum mismatch")
        value = pickle.loads(raw)
        size = len(raw)
        if size <= self.max_cache_bytes:
            while self.cache and (
                self.cache_bytes + size > self.max_cache_bytes or len(self.cache) >= 128
            ):
                _, (_, old_size) = self.cache.popitem(last=False)
                self.cache_bytes -= old_size
            self.cache[path] = value, size
            self.cache_bytes += size
        return value


class StudioCheckpointDict(MutableMapping):
    incremental = True

    def __init__(
        self,
        default_factory=None,
        *,
        filename,
        _backend=None,
        _path=(),
        allow_legacy=False,
    ):
        self.filename = str(filename)
        self.default_factory = default_factory
        self._path = _path
        if _backend is None:
            database = str(Path(filename).resolve()) + ".sqlite"
            with _registry_lock:
                _backend = _databases.get(database)
                if _backend is None:
                    _backend = _Database(database)
                    _databases[database] = _backend
        self.backend = _backend
        if not _path and Path(filename).is_file() and not allow_legacy:
            with self.backend.lock:
                ready = self.backend.conn.execute(
                    "SELECT value FROM metadata WHERE key='legacy_import_complete'"
                ).fetchone()
            if not ready:
                raise MigrationRequired(
                    "Studio 历史尚未迁移到按需存储；请在服务停止时运行迁移，原聊天文件保持完整。"
                )

    def _row(self, key):
        return self.backend.conn.execute(
            "SELECT kind,value FROM nodes WHERE path=?", (_encode((*self._path, key)),)
        ).fetchone()

    def __getitem__(self, key):
        with self.backend.lock:
            row = self._row(key)
            if row is None:
                if self.default_factory is None:
                    raise KeyError(key)
                self[key] = self.default_factory()
                row = self._row(key)
            path = (*self._path, key)
            if row[0] != "value":
                if row[0] == "value_ref":
                    identity = row[1].decode()
                    payload = self.backend.conn.execute(
                        "SELECT data FROM payloads WHERE id=?", (identity,)
                    ).fetchone()
                    if payload is None:
                        raise ValueError(
                            "Studio history payload missing; original history must be retained"
                        )
                    return self.backend.value(identity, payload[0])
                factory = pickle.loads(row[1]) if row[0] == "defaultdict" else None
                return type(self)(
                    factory, filename=self.filename, _backend=self.backend, _path=path
                )
            return self.backend.value(_encode(path), row[1])

    def __contains__(self, key):
        with self.backend.lock:
            return self._row(key) is not None

    def get(self, key, default=None):
        with self.backend.lock:
            return self[key] if key in self else default

    def setdefault(self, key, default=None):
        with self.backend.lock:
            if key not in self:
                self[key] = default
            return self[key]

    def _put(self, path, value):
        if isinstance(value, Mapping):
            factory = value.default_factory if isinstance(value, defaultdict) else None
            kind, data = ("defaultdict", _encode(factory)) if factory is not None else ("dict", b"")
        else:
            kind = "value_ref"
            raw = _encode(value)
            identity = sha256(raw).hexdigest()
            if (
                self.backend.conn.execute(
                    "SELECT 1 FROM payloads WHERE id=?", (identity,)
                ).fetchone()
                is None
            ):
                compressed = zlib.compress(raw, 1) if len(raw) >= 1024 else raw
                payload = b"Z" + compressed if len(compressed) < len(raw) else b"P" + raw
                self.backend.conn.execute("INSERT INTO payloads VALUES (?,?)", (identity, payload))
            data = identity.encode()
        self.backend.conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?)",
            (_encode(path), _encode(path[:-1]), _encode(path[-1]), kind, data),
        )
        if isinstance(value, Mapping):
            for key, item in value.items():
                self._put((*path, key), item)

    def __setitem__(self, key, value):
        with self.backend.lock, self.backend.conn:
            self.backend.invalidate()
            path = (*self._path, key)
            self._delete(path)
            self._put(path, value)

    def _delete(self, path):
        self.backend.conn.execute(
            """WITH RECURSIVE subtree(path) AS (
            SELECT path FROM nodes WHERE path=? UNION ALL
            SELECT n.path FROM nodes n JOIN subtree s ON n.parent=s.path
        ) DELETE FROM nodes WHERE path IN (SELECT path FROM subtree)""",
            (_encode(path),),
        )

    def __delitem__(self, key):
        with self.backend.lock, self.backend.conn:
            if key not in self:
                raise KeyError(key)
            self.backend.invalidate()
            self._delete((*self._path, key))

    def __iter__(self):
        with self.backend.lock:
            rows = self.backend.conn.execute(
                "SELECT key FROM nodes WHERE parent=? ORDER BY rowid",
                (_encode(self._path),),
            ).fetchall()
        return iter(pickle.loads(row[0]) for row in rows)

    def __len__(self):
        with self.backend.lock:
            return self.backend.conn.execute(
                "SELECT count(*) FROM nodes WHERE parent=?", (_encode(self._path),)
            ).fetchone()[0]

    def pop(self, key, default=_MISSING):
        with self.backend.lock:
            if key not in self:
                if default is _MISSING:
                    raise KeyError(key)
                return default
            value = self[key]
            if isinstance(value, StudioCheckpointDict):
                value = value.materialize()
            del self[key]
            return value

    def replace_from(self, replacement):
        """Retain lazy child views when upstream replaces the outer dictionary."""
        with self.backend.lock, self.backend.conn:
            keep = set(replacement)
            for key in list(self):
                if key not in keep:
                    del self[key]
            for key, value in replacement.items():
                if isinstance(value, StudioCheckpointDict):
                    if value.backend is self.backend and value._path == (
                        *self._path,
                        key,
                    ):
                        continue
                    value = value.materialize()
                self[key] = value

    def materialize(self):
        result = defaultdict(self.default_factory) if self.default_factory else {}
        for key in self:
            value = self[key]
            result[key] = value.materialize() if isinstance(value, StudioCheckpointDict) else value
        return result

    def clear(self):
        with self.backend.lock, self.backend.conn:
            self.backend.invalidate()
            if not self._path:
                self.backend.conn.execute("DELETE FROM nodes")
            else:
                for key in list(self):
                    self._delete((*self._path, key))

    def load(self):
        # Values are loaded individually by __getitem__, never as a whole store.
        return None

    def sync(self):
        with self.backend.lock:
            self.backend.conn.commit()

    def close(self):
        self.sync()
        self.backend.invalidate()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def migrate_studio_store(filename):
    """Offline conversion, retaining every original file and every leaf record."""
    filename = Path(filename)
    store = StudioCheckpointDict(filename=filename, allow_legacy=True)
    conn = store.backend.conn
    if conn.execute("SELECT value FROM metadata WHERE key='legacy_import_complete'").fetchone():
        return {"already_migrated": True, "source": str(filename)}
    fingerprint = filename.stat().st_size, filename.stat().st_mtime_ns
    with filename.open("rb") as stream:
        original = pickle.load(stream)
    if not isinstance(original, dict):
        raise ValueError("Unknown Studio checkpoint container; original left untouched")
    keys = list(original)
    store.clear()
    for key in keys:
        value = original.pop(key)
        store[key] = value

        def verify(expected, actual):
            if isinstance(expected, Mapping):
                if set(expected) != set(actual):
                    raise ValueError("Studio migration keys differ; original left untouched")
                for child_key in expected:
                    verify(expected[child_key], actual[child_key])
            elif expected != actual:
                raise ValueError("Studio migration value differs; original left untouched")

        verify(value, store[key])
        del value
        store.backend.invalidate()
    del original
    if fingerprint != (filename.stat().st_size, filename.stat().st_mtime_ns):
        raise RuntimeError("Studio history changed during migration; do not publish")
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Studio SQLite integrity check failed")
    with conn:
        conn.execute("INSERT INTO metadata VALUES ('legacy_import_complete','1')")
    store.sync()
    return {
        "verified": True,
        "top_level_records": len(keys),
        "before_bytes": fingerprint[0],
        "sqlite_bytes": sum(
            p.stat().st_size for p in store.backend.path.parent.glob(store.backend.path.name + "*")
        ),
    }


def checkpoint_container(*args, **kwargs):
    """Unmigrated projects keep their original history reader, with a diagnostic."""
    try:
        return StudioCheckpointDict(*args, **kwargs)
    except MigrationRequired:
        import logging
        from langgraph.checkpoint.memory import PersistentDict

        logging.getLogger(__name__).warning(
            "Studio 历史未离线迁移，暂用原兼容存储；聊天记录不会删除。"
        )
        return PersistentDict(*args, **kwargs)
