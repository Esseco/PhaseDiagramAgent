import hashlib
import sqlite3
import threading
import zlib

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde, STATE_FIELDS
from phase_agent.persistence.compact_checkpoints import CompressedSerde


def make(connection):
    return CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), connection, threading.RLock())


def test_streamed_list_matches_legacy_bytes_and_hash():
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        value = [{"id": i, "items": [None, True, 1.2], "name": "中", "payload": "x" * 100} for i in range(1400)]
        kind, expected = serde.raw_delegate.dumps_typed(value)
        identity = hashlib.sha256(kind.encode() + b"\0" + expected).hexdigest()
        encoded = serde._state({"rows": value})
        assert encoded[STATE_FIELDS]["rows"]["blob"] == identity
        stored_kind, stored = db.execute("select kind,data from checkpoint_payloads where id=?", (identity,)).fetchone()
        actual = zlib.decompress(stored) if stored_kind.startswith("zlib:") else stored
        assert actual == expected
        assert serde._restore_state(encoded)["rows"] == value


def test_interrupted_blob_write_rolls_back():
    class BrokenConnection:
        def __init__(self, connection): self.connection = connection
        @property
        def in_transaction(self): return self.connection.in_transaction
        def execute(self, *args, **kwargs): return self.connection.execute(*args, **kwargs)
        def rollback(self): return self.connection.rollback()
        def blobopen(self, *args, **kwargs): raise OSError("injected")
    with sqlite3.connect(":memory:") as db:
        serde = make(BrokenConnection(db))
        with pytest.raises(OSError, match="injected"):
            serde._state({"rows": [{"i": i, "payload": "x" * 100} for i in range(1400)]})
        assert db.execute("select count(*) from checkpoint_payloads").fetchone()[0] == 0


def test_uncommitted_blob_cannot_be_collected_concurrently(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    writer = sqlite3.connect(path)
    other = sqlite3.connect(path, timeout=0)
    try:
        serde = make(writer)
        value = [{"i": i, "payload": "x" * 100} for i in range(1400)]
        encoded = serde._state({"rows": value})
        assert writer.in_transaction
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("delete from checkpoint_payloads")
        writer.commit()
        assert serde._restore_state(encoded)["rows"] == value
    finally:
        writer.close()
        other.close()


def test_cross_process_resume_preserves_unsaved_data_and_does_not_replay(tmp_path):
    import subprocess
    import sys

    script = """
from pathlib import Path
import sys
from phase_agent.graphs.project.runner import run_search_workflow_graph
p = Path(sys.argv[1]); log = p / 'log'; flag = p / 'flag'; db = p / 'workflow.sqlite'
def record(s):
    with log.open('a') as f: f.write(s + '\\n')
def initialize():
    return {'_workflow_prepared': True, 'snapshot': {'config_version': 'v1'},
            'loaded_state': {'items': [{'i': i, 'payload': 'x' * 100} for i in range(1500)]}}
def collect(frame):
    record('collect'); frame['loaded_state']['pending'] = 'unsaved'; return frame
def analyze(frame):
    assert frame['loaded_state']['pending'] == 'unsaved'
    if not flag.exists():
        flag.write_text('yes'); raise RuntimeError('interrupted')
    return frame
def act(frame, **kwargs):
    record('action'); return frame
args = dict(checkpoint_path=db, invocation_id='one', initialize=initialize, collect=collect,
            analyze=analyze, wait=lambda f: f, assess=lambda f: f, act=act,
            finalize=lambda f: {'status': 'done', 'pending': f['loaded_state']['pending']})
try:
    print(run_search_workflow_graph(**args))
except RuntimeError as error:
    print('interrupted', str(error))
"""
    for _ in range(3):
        subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True)
    assert (tmp_path / "log").read_text().splitlines() == ["collect", "action"]


def test_rolled_back_blob_is_written_again():
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        value = [{"i": i, "payload": "x" * 100} for i in range(1400)]
        encoded = serde._state({"rows": value})
        db.rollback()
        encoded_again = serde._state({"rows": value})
        assert encoded == encoded_again
        assert serde._restore_state(encoded_again)["rows"] == value


def test_rolled_back_non_list_blob_is_written_again():
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        value = {"payload": "x" * 100000}
        encoded = serde._state({"field": value})
        db.rollback()
        encoded_again = serde._state({"field": value})
        assert encoded == encoded_again
        assert serde._restore_state(encoded_again)["field"] == value



def test_sealed_history_checks_sqlite_after_rollback_and_reencodes_local_change(monkeypatch):
    from phase_agent.graphs.business_checkpoint import checkpoint_frame, restore_frame

    state = {"rows": [{"value": i, "payload": "x" * 100} for i in range(1400)]}
    frame = {"loaded_state": state}
    archived = checkpoint_frame(frame, seal_lists=True)
    rows = archived["loaded_state"]["rows"]
    with pytest.raises(AttributeError):
        rows.append({"value": 0})
    restored = restore_frame({"loaded_state": {}}, archived)
    restored["loaded_state"]["rows"][0]["value"] = 123
    assert archived["loaded_state"]["rows"] == state["rows"]

    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        calls = []
        stream = serde._stream_list

        def track(value):
            calls.append(1)
            return stream(value)

        monkeypatch.setattr(serde, "_stream_list", track)
        pending = serde.dumps_typed(archived)
        regular = serde.dumps_typed({"channel_values": {"business_frame": archived}})
        assert len(calls) == 1
        assert serde.loads_typed(regular)["channel_values"]["business_frame"] == frame

        db.rollback()
        assert serde.dumps_typed(archived) == pending
        assert len(calls) == 2
        assert serde.loads_typed(pending)["loaded_state"]["rows"][0]["value"] == 0

        state["rows"][0]["value"] = 999
        changed = checkpoint_frame(frame, archived, seal_lists=True)
        assert serde.loads_typed(serde.dumps_typed(changed)) == frame
        assert len(calls) == 3
        assert serde.loads_typed(pending)["loaded_state"]["rows"][0]["value"] == 0



def test_sealed_reference_is_protected_until_checkpoint_write(tmp_path):
    from contextlib import closing
    from phase_agent.graphs.business_checkpoint import checkpoint_frame

    frame = checkpoint_frame(
        {"loaded_state": {"rows": [{"payload": "x" * 100} for _ in range(1400)]}},
        seal_lists=True,
    )
    path = tmp_path / "checkpoint.sqlite"
    with closing(sqlite3.connect(path)) as writer, closing(
        sqlite3.connect(path, timeout=0)
    ) as other:
        serde = make(writer)
        pending = serde.dumps_typed(frame)
        writer.commit()
        assert serde.dumps_typed(frame) == pending
        assert writer.in_transaction
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("DELETE FROM checkpoint_payloads")
        writer.commit()
        assert serde.loads_typed(pending)["loaded_state"]["rows"][0]["payload"] == "x" * 100



def test_sealed_child_start_preserves_existing_inline_encoding():
    from phase_agent.graphs.business_checkpoint import checkpoint_frame

    frame = {"loaded_state": {"rows": [{"value": i} for i in range(1400)]}}
    sealed = checkpoint_frame(frame, seal_lists=True)
    start = {"channel_values": {"__start__": {"business_frame": sealed}}}
    previous = {"channel_values": {"__start__": {"business_frame": frame}}}
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        encoded = serde.dumps_typed(start)
        assert encoded == serde.dumps_typed(previous)
        assert serde.loads_typed(encoded) == previous



def test_routing_read_defers_business_blobs_and_full_read_still_validates():
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        frame = {"loaded_state": {"large": "x" * 100000}, "snapshot": {"config_version": "v1"}}
        value = {"channel_values": {
            "request_id": "p", "business_frame": frame,
            "completed_response": {"status": "completed", "state": frame["loaded_state"]},
        }}
        encoded = serde.dumps_typed(value)
        statements = []
        db.set_trace_callback(statements.append)
        with serde.routing_only():
            header = serde.loads_typed(encoded)["channel_values"]
        assert header == {
            "request_id": "p", "business_frame": {"loaded_state": {}, "snapshot": {"config_version": "v1"}},
            "completed_response": {"status": "completed", "state": {}},
        }
        assert not any("checkpoint_payloads" in statement for statement in statements)
        assert serde.loads_typed(encoded) == value
        db.execute("DELETE FROM checkpoint_payloads")
        with pytest.raises(ValueError, match="Checkpoint payload missing"):
            serde.loads_typed(encoded)



def test_wait_checkpoint_reads_shared_history_once_and_other_reads_stay_mutable():
    with sqlite3.connect(":memory:") as db:
        serde = make(db)
        left = [{"value": i, "payload": "x" * 100} for i in range(1400)]
        right = [dict(row) for row in left]
        value = {
            "id": "native-wait",
            "channel_values": {
                "business_frame": {"loaded_state": {"rows": left}},
                "completed_response": {"state": {"rows": right}},
            },
        }
        encoded = serde.dumps_typed(value)
        statements = []
        db.set_trace_callback(statements.append)
        with serde.share_wait_history("native-wait"):
            read = serde.loads_typed(encoded)["channel_values"]
        reads = [s for s in statements if "SELECT kind,data FROM checkpoint_payloads" in s]
        assert len(reads) == 1
        a = read["business_frame"]["loaded_state"]["rows"]
        b = read["completed_response"]["state"]["rows"]
        assert a is b and a == left
        with pytest.raises(AttributeError):
            a.append({})
        a.mutable_copy()[0]["value"] = -1
        assert b == right
        with serde.share_wait_history("different-checkpoint"):
            normal = serde.loads_typed(encoded)["channel_values"]
        normal["business_frame"]["loaded_state"]["rows"][0]["value"] = -1
        assert normal["completed_response"]["state"]["rows"] == right
        db.execute("UPDATE checkpoint_payloads SET kind='msgpack', data=X'90'")
        with serde.share_wait_history("native-wait"), pytest.raises(ValueError, match="checksum"):
            serde.loads_typed(encoded)
