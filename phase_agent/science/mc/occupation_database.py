"""Project-owned persistent branch occupancy index."""

import base64
from pathlib import Path
import sqlite3
from Process_AL_MC.occupation_history import encode_snapshot, decode_snapshot


class OccupationHistory:
    """Indexed exact bitmaps for one branch, optionally in a shared project DB."""

    def __init__(self, path, scope, site_count, basis, *, seed=None):
        if not isinstance(scope, str) or not scope or len(scope) > 256:
            raise ValueError("MC occupancy history requires an explicit branch scope")
        if not isinstance(site_count, int) or isinstance(site_count, bool) or site_count <= 0:
            raise ValueError("MC occupancy history requires positive Na site count")
        self.scope, self.site_count, self.basis = scope, site_count, basis
        self.width = (site_count + 7) // 8
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS branches(scope TEXT PRIMARY KEY, n INTEGER, basis TEXT);"
            "CREATE TABLE IF NOT EXISTS seen(scope TEXT, bits BLOB, PRIMARY KEY(scope,bits)) WITHOUT ROWID;"
            "CREATE TABLE IF NOT EXISTS imports(source TEXT PRIMARY KEY, checksum TEXT);"
        )
        existing = self.db.execute(
            "SELECT n,basis FROM branches WHERE scope=?", (scope,)
        ).fetchone()
        if existing and existing != (site_count, basis):
            self.close()
            raise ValueError(
                "Branch Na site ordering changed; occupancy histories cannot be combined"
            )
        self.db.execute("INSERT OR IGNORE INTO branches VALUES(?,?,?)", (scope, site_count, basis))
        self.db.commit()
        self.new_records = set()
        if seed:
            self.merge(base64.b64decode(seed, validate=True), track_new=False)
        self.count = self.db.execute(
            "SELECT count(*) FROM seen WHERE scope=?", (scope,)
        ).fetchone()[0]

    def _validate(self, bits):
        if not isinstance(bits, bytes) or len(bits) != self.width:
            raise ValueError("MC occupancy bitmap does not match Na site count")
        if self.site_count % 8 and bits[-1] >> (self.site_count % 8):
            raise ValueError("MC occupancy bitmap has nonzero padding")

    def __contains__(self, bits):
        self._validate(bits)
        return (
            self.db.execute(
                "SELECT 1 FROM seen WHERE scope=? AND bits=?", (self.scope, bits)
            ).fetchone()
            is not None
        )

    def __len__(self):
        return self.count

    def add(self, bits):
        self._validate(bits)
        cursor = self.db.execute("INSERT OR IGNORE INTO seen VALUES(?,?)", (self.scope, bits))
        self.db.commit()
        if cursor.rowcount:
            self.count += 1
            self.new_records.add(bits)
        return bool(cursor.rowcount)

    def merge(self, data, *, track_new=False):
        header, records = decode_snapshot(data)
        if header != {"scope": self.scope, "site_count": self.site_count, "basis": self.basis}:
            raise ValueError("MC occupancy snapshot branch or site ordering mismatch")
        for bits in records:
            self._validate(bits)
        previous = self.db.total_changes
        self.db.executemany(
            "INSERT OR IGNORE INTO seen VALUES(?,?)", [(self.scope, b) for b in records]
        )
        self.db.commit()
        added = self.db.total_changes - previous
        self.count = self.db.execute(
            "SELECT count(*) FROM seen WHERE scope=?", (self.scope,)
        ).fetchone()[0]
        if track_new:
            self.new_records.update(records)
        return added

    def snapshot(self, *, delta=False):
        records = (
            self.new_records
            if delta
            else [
                row[0]
                for row in self.db.execute(
                    "SELECT bits FROM seen WHERE scope=? ORDER BY bits", (self.scope,)
                )
            ]
        )
        return encode_snapshot(self.scope, self.site_count, self.basis, records)

    def seed(self):
        return base64.b64encode(self.snapshot()).decode("ascii")

    def write_delta(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(self.snapshot(delta=True))
        temporary.replace(path)

    def imported(self, source, checksum):
        return (
            self.db.execute(
                "SELECT 1 FROM imports WHERE source=? AND checksum=?", (source, checksum)
            ).fetchone()
            is not None
        )

    def mark_imported(self, source, checksum):
        self.db.execute("INSERT OR REPLACE INTO imports VALUES(?,?)", (source, checksum))
        self.db.commit()

    def close(self):
        self.db.close()
