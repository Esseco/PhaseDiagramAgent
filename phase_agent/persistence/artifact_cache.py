"""Process-local caches for immutable artifacts, keyed by file metadata.

Returned results and recorded task files are never rewritten in place by the
workflow; a replaced file gets a new size/mtime/inode, so the key changes and
the cached value is recomputed. Missing files are never cached.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from threading import Lock

_lock = Lock()
_digests: OrderedDict = OrderedDict()
_MAX_ENTRIES = 8192


def stat_key(path):
    """Return a metadata identity for an existing file, or None when absent."""
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return None
    return (str(source.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)


def file_sha256(path):
    """SHA-256 of a file, reused while its metadata identity is unchanged."""
    key = stat_key(path)
    if key is not None:
        with _lock:
            cached = _digests.get(key)
            if cached is not None:
                _digests.move_to_end(key)
                return cached
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if key is not None and stat_key(path) == key:
        with _lock:
            _digests[key] = digest
            while len(_digests) > _MAX_ENTRIES:
                _digests.popitem(last=False)
    return digest
